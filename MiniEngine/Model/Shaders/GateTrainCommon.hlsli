// File: GateTrainCommon.hlsli

// =========================================================================
//   Thread Group Sizes
// =========================================================================

#define BACKPROP_THREADGROUP_SIZE 64
#define OPTIMIZATION_THREADGROUP_SIZE 64

// =========================================================================
//   Network Configuration (8 -> 16 -> 4)
// =========================================================================

#define LAYER_COUNT 3
#define INPUT_LAYER 0
#define HIDDEN_LAYER 1
#define OUTPUT_LAYER 2

#define MAX_NEURON_QUARTETS_PER_LAYER 4 // Max 16 neurons / 4
#define ACTIVATION_QUARTETS_PER_NETWORK (2 + 4 + 1) // 8 inputs(2) + 16 hidden(4) + 4 outputs(1)

#define LEAKY_RELU_SLOPE 0.01f
#define FLOAT4_PACKING_CONSTANT 16384.0f // Scale for fixed-point atomic addition

// =========================================================================
//   Structs
// =========================================================================

struct AdamData
{
    float4 mean, variance;
    uint stepCount;
    uint3 pad;
};

struct GlobalTriangle
{
    uint i0, i1, i2, materialIdx;
};

struct GateFeature
{
    float4 data[2];
};

struct GateEncodingData
{
    float3 barycentrics;
    uint triangleId;
};

struct MeshColorSample
{
    uint3 indices; // Local 1D indices [0, K-1]
    float3 weights; // Local barycentrics
};

// =========================================================================
//   Resources
// =========================================================================

cbuffer RootConstantsCB : register(b0)
{
    uint trainingStep;
    uint totalTriangles;
    float featureLearningRate;
    float mlpLearningRate;
    float adamEpsilon;
    float adamBeta1;
    float adamBeta2;
    float weightDecay;
    float screenSpaceRatio;
    uint VertexStride;
    uint uvOffset;
    uint screenWidth;
    uint screenHeight;
    // New Mesh Color variables
    uint meshColorR;
    uint meshColorK;
    uint totalUniqueMeshColorPoints;
    uint totalDuplicatedMeshColorPoints;
};

#ifdef GATE_INFERENCE
// -------------------------------------------------------------------------
//   INFERENCE RESOURCES
// -------------------------------------------------------------------------

StructuredBuffer<GateFeature> FeatureBuffer      : register(t0);
StructuredBuffer<float4>      MLPParameterBuffer : register(t1);

#else
// -------------------------------------------------------------------------
//   TRAINING RESOURCES
// -------------------------------------------------------------------------

// --- SRVs (Read-Only) ---
StructuredBuffer<GlobalTriangle> GlobalTriangleBuffer  : register(t0);
ByteAddressBuffer                VertexUVBuffer        : register(t1);
Texture2D<uint>                  VisibilityBuffer      : register(t2, space0);
StructuredBuffer<uint>           VertexMappingBuffer   : register(t3); // TotalVertexID -> UniqueVertexID
StructuredBuffer<GateFeature>    UniqueFeatureBuffer   : register(t4); // Unique features for Backprop
StructuredBuffer<GlobalTriangle> SpatialTriangleBuffer : register(t5); // Welded triangles for Network

SamplerState                     LinearSampler         : register(s0);
Texture2D<float4>                BindlessTextures[]    : register(t0, space1);

// --- UAVs (Read/Write) ---
// Features
RWStructuredBuffer<GateFeature>  DuplicatedFeatureBuffer : register(u0);
RWStructuredBuffer<int4>         FeatureGradientBuffer   : register(u1);
RWStructuredBuffer<AdamData>     FeatureAdamBuffer       : register(u2);

// MLP
RWStructuredBuffer<float4>       MLPParameterBuffer      : register(u3);
RWStructuredBuffer<int4>         MLPGradientBuffer       : register(u4);
RWStructuredBuffer<AdamData>     MLPAdamBuffer           : register(u5);
#endif

// =========================================================================
//   Helpers: RNG & Float Packing
// =========================================================================

uint pcgHash(uint v)
{
    const uint state = v * 747796405u + 2891336453u;
    const uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float rand(inout uint rngState)
{
    rngState ^= rngState << 13;
    rngState ^= rngState >> 17;
    rngState ^= rngState << 5;
    return asfloat(0x3f800000 | (rngState >> 9)) - 1.0f;
}

float4 unpackFloat4(int4 x)
{
    return float4(x) / FLOAT4_PACKING_CONSTANT;
}

int4 packFloat4(float4 x)
{
    return int4(x * FLOAT4_PACKING_CONSTANT);
}

// =========================================================================
//   Activations (Shared)
// =========================================================================

float4 activationFunction(float4 v)
{
    return float4(
        (v.x >= 0.0f) ? v.x : (v.x * LEAKY_RELU_SLOPE),
        (v.y >= 0.0f) ? v.y : (v.y * LEAKY_RELU_SLOPE),
        (v.z >= 0.0f) ? v.z : (v.z * LEAKY_RELU_SLOPE),
        (v.w >= 0.0f) ? v.w : (v.w * LEAKY_RELU_SLOPE)
    );
}

float4 activationFunctionDeriv(float4 v)
{
    return float4(
        (v.x <= 0.0f) ? LEAKY_RELU_SLOPE : 1.0f,
        (v.y <= 0.0f) ? LEAKY_RELU_SLOPE : 1.0f,
        (v.z <= 0.0f) ? LEAKY_RELU_SLOPE : 1.0f,
        (v.w <= 0.0f) ? LEAKY_RELU_SLOPE : 1.0f
    );
}

float4 activationFunctionOutput(float4 v)
{
    return 1.0f / (1.0f + exp(-v));
}

float4 activationFunctionOutputDeriv(float4 v)
{
    return v * (1.0f - v);
}


// =========================================================================
//   Mesh Colors: Yuksel's 2D -> 1D Filter
// =========================================================================

// Converts integer sub-triangle coordinates (i, j) to a linear 1D index
uint GetMeshColor1DIndex(uint i, uint j, uint R)
{
    return i * (R + 1) - (i * (i - 1)) / 2 + j;
}

MeshColorSample GetMeshColorSample(float3 barycentrics, uint R)
{
    // 1. Scale barycentrics by resolution
    float u_prime = barycentrics.x * R;
    float v_prime = barycentrics.y * R;

    // 2. Find integer anchor of the sub-triangle
    uint i = (uint) floor(u_prime);
    uint j = (uint) floor(v_prime);

    // Handle floating point inaccuracies exactly on the V0 or V1 corners
    if (i >= R)
    {
        i = R;
        j = 0;
    }
    if (j >= R)
    {
        i = 0;
        j = R;
    }

    // 3. Find fractional offsets
    float du = u_prime - (float) i;
    float dv = v_prime - (float) j;

    MeshColorSample sample;

    // 4. Determine which half of the quad we are in
    if (du + dv <= 1.0f)
    {
        sample.indices.x = GetMeshColor1DIndex(i + 1, j, R); // +u direction
        sample.indices.y = GetMeshColor1DIndex(i, j + 1, R); // +v direction
        sample.indices.z = GetMeshColor1DIndex(i, j, R); // Origin
        sample.weights = float3(du, dv, 1.0f - du - dv);
    }
    else
    {
        sample.indices.x = GetMeshColor1DIndex(i + 1, j + 1, R); // Opposite corner
        sample.indices.y = GetMeshColor1DIndex(i, j + 1, R); // +v direction
        sample.indices.z = GetMeshColor1DIndex(i + 1, j, R); // +u direction
        sample.weights = float3(du + dv - 1.0f, 1.0f - du, 1.0f - dv);
    }

    return sample;
}

// =========================================================================
//   TRAINING ONLY FUNCTIONS
// =========================================================================

#ifndef GATE_INFERENCE

void accumulateGradient(RWStructuredBuffer<int4> gradientTarget, const uint gradientIndex, float4 gradient)
{
    // Clip gradients to prevent exploding loss
    gradient = clamp(gradient, -0.5f, 0.5f);
    const int4 packed = packFloat4(gradient);
    InterlockedAdd(gradientTarget[gradientIndex].x, packed.x);
    InterlockedAdd(gradientTarget[gradientIndex].y, packed.y);
    InterlockedAdd(gradientTarget[gradientIndex].z, packed.z);
    InterlockedAdd(gradientTarget[gradientIndex].w, packed.w);
}

void gateEncoding(const GateEncodingData gateData, inout uint activationIndex, inout float4 activations[ACTIVATION_QUARTETS_PER_NETWORK])
{
    // Find our 3 local points and weights
    MeshColorSample mc = GetMeshColorSample(gateData.barycentrics, meshColorR);

    // Calculate global offset for this specific triangle in the DUPLICATED buffer
    uint baseIdx = gateData.triangleId * meshColorK;

    // Fast, Cache-Coherent Read!
    GateFeature f0 = DuplicatedFeatureBuffer[baseIdx + mc.indices.x];
    GateFeature f1 = DuplicatedFeatureBuffer[baseIdx + mc.indices.y];
    GateFeature f2 = DuplicatedFeatureBuffer[baseIdx + mc.indices.z];

    activations[activationIndex++] = mc.weights.x * f0.data[0] + mc.weights.y * f1.data[0] + mc.weights.z * f2.data[0];
    activations[activationIndex++] = mc.weights.x * f0.data[1] + mc.weights.y * f1.data[1] + mc.weights.z * f2.data[1];
}

void gateEncodingBackprop(const GateEncodingData gateData, inout float4 errors[ACTIVATION_QUARTETS_PER_NETWORK])
{
    MeshColorSample mc = GetMeshColorSample(gateData.barycentrics, meshColorR);
    uint baseIdx = gateData.triangleId * meshColorK;

    // Use the N:M Mapping Buffer to figure out where to accumulate gradients in the UNIQUE buffer!
    uint unique0 = VertexMappingBuffer[baseIdx + mc.indices.x]; // Note: Keep using t3 register, just renamed in C++
    uint unique1 = VertexMappingBuffer[baseIdx + mc.indices.y];
    uint unique2 = VertexMappingBuffer[baseIdx + mc.indices.z];

    float4 inputGrad0 = errors[0];
    float4 inputGrad1 = errors[1];

    accumulateGradient(FeatureGradientBuffer, unique0 * 2 + 0, inputGrad0 * mc.weights.x);
    accumulateGradient(FeatureGradientBuffer, unique0 * 2 + 1, inputGrad1 * mc.weights.x);

    accumulateGradient(FeatureGradientBuffer, unique1 * 2 + 0, inputGrad0 * mc.weights.y);
    accumulateGradient(FeatureGradientBuffer, unique1 * 2 + 1, inputGrad1 * mc.weights.y);

    accumulateGradient(FeatureGradientBuffer, unique2 * 2 + 0, inputGrad0 * mc.weights.z);
    accumulateGradient(FeatureGradientBuffer, unique2 * 2 + 1, inputGrad1 * mc.weights.z);
}

void evalLayerActivations(inout float4 activations[ACTIVATION_QUARTETS_PER_NETWORK], uint weightOffset, uint prevNeuronOffset, uint currNeuronOffset, uint currQuartets, uint prevQuartets, uint layerType)
{
    for (uint q = 0; q < currQuartets; q++)
    {
        float4 neuronValue = 0.0f;
        for (uint prevQ = prevNeuronOffset; prevQ < prevNeuronOffset + prevQuartets; prevQ++)
        {
            const float4 prevAct = activations[prevQ];
            neuronValue.x += dot(MLPParameterBuffer[weightOffset++], prevAct);
            neuronValue.y += dot(MLPParameterBuffer[weightOffset++], prevAct);
            neuronValue.z += dot(MLPParameterBuffer[weightOffset++], prevAct);
            neuronValue.w += dot(MLPParameterBuffer[weightOffset++], prevAct);
        }
        const float4 bias = MLPParameterBuffer[weightOffset++];
        activations[currNeuronOffset++] = (layerType == HIDDEN_LAYER) ? activationFunction(neuronValue + bias) : activationFunctionOutput(neuronValue + bias);
    }
}

void backpropLayer(const float3 target, inout float4 activations[ACTIVATION_QUARTETS_PER_NETWORK], inout float4 errors[ACTIVATION_QUARTETS_PER_NETWORK], uint prevLayerQuartets, uint currLayerQuartets, uint prevOffset, uint currOffset, uint weightIndex, uint layerType)
{
    // Initialize the previous layer's error to 0 so we can accumulate the transposed weights into it
    for (uint pq = prevOffset; pq < prevOffset + prevLayerQuartets; pq++)
        errors[pq] = 0.0f;

    for (uint q = currOffset; q < currOffset + currLayerQuartets; q++)
    {
        const float4 act = activations[q];
        float4 dCost_O = (layerType == OUTPUT_LAYER) ? (act - float4(target, 0.0f)) : errors[q];
        const float4 dCost_Z = dCost_O * ((layerType == HIDDEN_LAYER) ? activationFunctionDeriv(act) : activationFunctionOutputDeriv(act));
        
        // Weights Gradient & Error Backprop
        for (uint prevQ = prevOffset; prevQ < prevOffset + prevLayerQuartets; prevQ++)
        {
            const float4 prevAct = activations[prevQ];

            // Load the 4 weight vectors for these 4 neurons
            float4 wX = MLPParameterBuffer[weightIndex];
            float4 wY = MLPParameterBuffer[weightIndex + 1];
            float4 wZ = MLPParameterBuffer[weightIndex + 2];
            float4 wW = MLPParameterBuffer[weightIndex + 3];

            // Backpropagate error to the previous layer (Matrix Transpose equivalent)
            errors[prevQ] += wX * dCost_Z.x + wY * dCost_Z.y + wZ * dCost_Z.z + wW * dCost_Z.w;

            // Accumulate weight gradients
            accumulateGradient(MLPGradientBuffer, weightIndex++, dCost_Z.x * prevAct);
            accumulateGradient(MLPGradientBuffer, weightIndex++, dCost_Z.y * prevAct);
            accumulateGradient(MLPGradientBuffer, weightIndex++, dCost_Z.z * prevAct);
            accumulateGradient(MLPGradientBuffer, weightIndex++, dCost_Z.w * prevAct);
        }
        // Bias Gradient
        accumulateGradient(MLPGradientBuffer, weightIndex++, dCost_Z);
    }
}

// Update signature to include currentValue, lr, and wd
float4 ApplyAdam(float4 gradient, float4 currentValue, inout AdamData adamData, float lr, float wd)
{
    adamData.stepCount += 1;

    float localBeta1T = pow(adamBeta1, (float)adamData.stepCount);
    float localBeta2T = pow(adamBeta2, (float)adamData.stepCount);

    adamData.mean = lerp(gradient, adamData.mean, adamBeta1);
    adamData.variance = lerp(gradient * gradient, adamData.variance, adamBeta2);
    
    float4 correctedMean = adamData.mean / (1.0f - localBeta1T);
    float4 correctedVariance = adamData.variance / (1.0f - localBeta2T);
    
    // 1. Standard Adam step
    float4 adamStep = correctedMean * rsqrt(correctedVariance + adamEpsilon);
    
    // 2. Decoupled Weight Decay (AdamW)
    float4 decayStep = currentValue * wd;

    // 3. Apply learning rate to both
    return -lr * (adamStep + decayStep);
}
#endif // !GATE_INFERENCE

// =========================================================================
//   MLP Forward Layer (Inference / Optimized)
// =========================================================================

void evalLayer(inout float4 previousActivations[MAX_NEURON_QUARTETS_PER_LAYER], inout float4 currentActivations[MAX_NEURON_QUARTETS_PER_LAYER],
    uint paramOffset, const uint neuronQuartetCountCurrentLayer, const uint neuronQuartetCountPreviousLayer, const uint layerType)
{
    for (uint neuronQuartet = 0; neuronQuartet < neuronQuartetCountCurrentLayer; neuronQuartet++)
    {
        float4 neuronValue = 0.0f;
        
        for (uint previousNeuronQuartet = 0; previousNeuronQuartet < neuronQuartetCountPreviousLayer; previousNeuronQuartet++)
        {
            const float4 prevAct = previousActivations[previousNeuronQuartet];
            neuronValue.x += dot(MLPParameterBuffer[paramOffset++], prevAct);
            neuronValue.y += dot(MLPParameterBuffer[paramOffset++], prevAct);
            neuronValue.z += dot(MLPParameterBuffer[paramOffset++], prevAct);
            neuronValue.w += dot(MLPParameterBuffer[paramOffset++], prevAct);
        }
        
        const float4 bias = MLPParameterBuffer[paramOffset++];
        currentActivations[neuronQuartet] = (layerType == HIDDEN_LAYER) ? activationFunction(neuronValue + bias) : currentActivations[neuronQuartet] = activationFunctionOutput(neuronValue + bias);
    }
}
