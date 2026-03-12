// File: GateTrainCommon.hlsli

// =========================================================================
//   Thread Group Sizes
// =========================================================================
#define BACKPROP_THREADGROUP_SIZE 64
#define OPTIMIZATION_THREADGROUP_SIZE 64

// =========================================================================
//   Shared Structs
// =========================================================================
struct AdamData
{
    float4 mean;
    float4 variance;
};

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
//   Resources
// =========================================================================

cbuffer RootConstantsCB : register(b0)
{
    uint trainingStep;
    uint totalTriangles;
    float learningRate;
    float adamEpsilon;
    float adamBeta1;
    float adamBeta2;
    float adamBeta1T; // Beta1^t
    float adamBeta2T; // Beta2^t
    uint VertexStride;
    uint uvOffset;
};

// Sponza Geometry & Target
struct GlobalTriangle
{
    uint i0;
    uint i1;
    uint i2;
    uint materialIdx;
};

StructuredBuffer<GlobalTriangle> TriangleBuffer : register(t0);
ByteAddressBuffer VertexUVBuffer : register(t1);
Texture2D<float4> BindlessTextures[] : register(t0, space1);
SamplerState LinearSampler : register(s0);

// GATE Features (Per-Vertex)
struct GateFeature
{
    float4 data[2];
};
RWStructuredBuffer<GateFeature> GateFeatureBuffer : register(u0);
RWStructuredBuffer<int4> GateFeatureGradientBuffer : register(u1);
RWStructuredBuffer<AdamData> GateFeatureAdamBuffer : register(u2);

// MLP Parameters
RWStructuredBuffer<float4> MLPParameterBuffer : register(u3);
RWStructuredBuffer<int4> MLPGradientBuffer : register(u4);
RWStructuredBuffer<AdamData> MLPAdamBuffer : register(u5);

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

// =========================================================================
//   Activations
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

// Output uses Sigmoid to clamp color 0.0 -> 1.0
float4 activationFunctionOutput(float4 v)
{
    return 1.0f / (1.0f + exp(-v));
}
float4 activationFunctionOutputDeriv(float4 v)
{
    return v * (1.0f - v);
}

// =========================================================================
//   GATE Encoding (Barycentric Interpolation)
// =========================================================================

struct GateEncodingData
{
    float3 barycentrics;
    uint3 indices;
};

void gateEncoding(const GateEncodingData gateData, inout uint activationIndex, inout float4 activations[ACTIVATION_QUARTETS_PER_NETWORK])
{
    GateFeature f0 = GateFeatureBuffer[gateData.indices.x];
    GateFeature f1 = GateFeatureBuffer[gateData.indices.y];
    GateFeature f2 = GateFeatureBuffer[gateData.indices.z];

    activations[activationIndex++] = gateData.barycentrics.x * f0.data[0] + gateData.barycentrics.y * f1.data[0] + gateData.barycentrics.z * f2.data[0];
    activations[activationIndex++] = gateData.barycentrics.x * f0.data[1] + gateData.barycentrics.y * f1.data[1] + gateData.barycentrics.z * f2.data[1];
}

void gateEncodingBackprop(const GateEncodingData gateData, inout float4 errors[ACTIVATION_QUARTETS_PER_NETWORK])
{
    float4 inputGrad0 = errors[0];
    float4 inputGrad1 = errors[1];

    accumulateGradient(GateFeatureGradientBuffer, gateData.indices.x * 2 + 0, inputGrad0 * gateData.barycentrics.x);
    accumulateGradient(GateFeatureGradientBuffer, gateData.indices.x * 2 + 1, inputGrad1 * gateData.barycentrics.x);

    accumulateGradient(GateFeatureGradientBuffer, gateData.indices.y * 2 + 0, inputGrad0 * gateData.barycentrics.y);
    accumulateGradient(GateFeatureGradientBuffer, gateData.indices.y * 2 + 1, inputGrad1 * gateData.barycentrics.y);

    accumulateGradient(GateFeatureGradientBuffer, gateData.indices.z * 2 + 0, inputGrad0 * gateData.barycentrics.z);
    accumulateGradient(GateFeatureGradientBuffer, gateData.indices.z * 2 + 1, inputGrad1 * gateData.barycentrics.z);
}

// =========================================================================
//   MLP Forward / Backward Layers
// =========================================================================

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
    {
        errors[pq] = 0.0f;
    }

    for (uint q = currOffset; q < currOffset + currLayerQuartets; q++)
    {
        const float4 act = activations[q];
        float4 dCost_O = 0.0f;

        if (layerType == OUTPUT_LAYER)
        {
            dCost_O = (act - float4(target, 0.0f)); // L2 Loss Derivative
        }
        else
        {
            dCost_O = errors[q]; // Read accumulated errors from the layer above
        }

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

float4 ApplyAdam(float4 gradient, inout AdamData adamData)
{
    adamData.mean = lerp(gradient, adamData.mean, adamBeta1);
    adamData.variance = lerp(gradient * gradient, adamData.variance, adamBeta2);
    float4 correctedMean = adamData.mean / (1.0f - adamBeta1T);
    float4 correctedVariance = adamData.variance / (1.0f - adamBeta2T);
    return -learningRate * (correctedMean * rsqrt(correctedVariance + adamEpsilon));
}