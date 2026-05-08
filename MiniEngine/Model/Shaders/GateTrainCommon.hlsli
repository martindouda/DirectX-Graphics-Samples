// =========================================================================================
// File: GateTrainCommon.hlsli
//
// ATTRIBUTION NOTICE:
// A significant portion of the neural network mathematics (forward pass, backpropagation, 
// and AdamW optimizer) in this file is derived and adapted from the MLPZen framework 
// created by Jakub Bokšanský.
//
// Original source: https://github.com/boksajak/MLPZen/blob/master/src/shaders/MLPZen.hlsl
// =========================================================================================

// Thread group sizes
#define BACKPROP_THREADGROUP_SIZE 1024
#define OPTIMIZATION_MLP_THREADGROUP_SIZE 64
#define OPTIMIZATION_FEATURES_THREADGROUP_SIZE 1024
#define BROADCAST_THREADGROUP_SIZE 1024

// Network layout
#define LAYER_COUNT 3
#define INPUT_LAYER 0
#define HIDDEN_LAYER 1
#define OUTPUT_LAYER 2

// Max memory bounds for dynamic arrays (8 quartets = 32 floats max)
#define MAX_FEATURE_QUARTETS 8                      
#define MAX_NEURON_QUARTETS_PER_LAYER 4             
#define ACTIVATION_QUARTETS_PER_NETWORK (MAX_FEATURE_QUARTETS + 4 + 1) 

// Hyperparameters and fixed-point math scaling
#define LEAKY_RELU_SLOPE 0.01f
#define FLOAT4_PACKING_CONSTANT 16384.0f          

struct AdamData
{
    float4 mean, variance;
    uint stepCount;
    uint3 pad;
};

struct GlobalTriangle
{
    uint i0, i1, i2, materialIdx;
    uint pointOffset;
    uint resolution;
    uint pointsPerTri;
    uint pad;
};

struct GateEncodingData
{
    float3 barycentrics;
    uint3 indices;
};

#ifndef GATE_INFERENCE
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
    uint totalMeshColorPoints;
    float aoRadius;
    uint uniqueVertexCount;
    float3 sunDirection;
    uint featureFloats;
    uint featureQuartets;
    uint mlpQuartets;
    uint learningMode;
    float maxGradientClip;
    uint aoSamples;
    uint shadowSamples;
    float shadowSoftnessAngle;
    float pad;
};
#else
cbuffer MeshConstants : register(b1) 
{ 
    uint globalTriangleOffset; 
    uint lightingMode;
    uint renderFlags;
    uint materialIdx;
    float3 sunDirection;
    float sunIntensity;
    uint featureFloats;
    uint featureQuartets; 
    uint aoSamples;
    uint shadowSamples;
    float3 cameraPos;
    float aoRadius;
    float shadowSoftnessAngle;
    uint frameIndex;
    float2 pad;
};
#endif

#ifdef GATE_INFERENCE
StructuredBuffer<float4>            FeatureBuffer           : register(t0);
StructuredBuffer<float4>            MLPParameterBuffer      : register(t1);
StructuredBuffer<GlobalTriangle>    GlobalTriangleBuffer    : register(t2);
RaytracingAccelerationStructure     SceneBVH                : register(t3);

Texture2D<float4>                   BindlessTextures[]      : register(t0, space1);
SamplerState                        LinearSampler           : register(s0);
#else
// --- SRVs (t0 - t7) ---
StructuredBuffer<GlobalTriangle> GlobalTriangleBuffer : register(t0);
ByteAddressBuffer VertexUVBuffer : register(t1);
StructuredBuffer<uint> VertexMappingBuffer : register(t2);
StructuredBuffer<float4> UniqueFeatureBuffer : register(t3);
RaytracingAccelerationStructure SceneBVH : register(t4);
StructuredBuffer<uint> UniqueToDuplicateOffsetBuffer : register(t5);
StructuredBuffer<uint> UniqueToDuplicateCountBuffer : register(t6);
StructuredBuffer<uint> DuplicateIndicesBuffer : register(t7);

// --- Descriptor Tables (t8, t0 space1) ---
Texture2D<uint> VisibilityBuffer : register(t8, space0);
Texture2D<float4> BindlessTextures[] : register(t0, space1);
SamplerState LinearSampler : register(s0);

// --- UAVs (u0 - u7) ---
RWStructuredBuffer<float4> TargetFeatureBufferUAV : register(u0);
RWStructuredBuffer<int4> FeatureGradientBuffer : register(u1);
RWStructuredBuffer<AdamData> FeatureAdamBuffer : register(u2);
RWStructuredBuffer<float4> MLPParameterBuffer : register(u3);
RWStructuredBuffer<int4> MLPGradientBuffer : register(u4);
RWStructuredBuffer<AdamData> MLPAdamBuffer : register(u5);
RWByteAddressBuffer LossBuffer : register(u6);
RWStructuredBuffer<uint> FeatureDirtyBuffer : register(u7);
#endif

// Uniformly samples a direction within a cone of angle 'coneAngle' (in radians) around 'dir'
float3 getConeSample(float u1, float u2, float3 dir, float coneAngle)
{
    float cosTheta = cos(coneAngle);
    float z = cosTheta + u1 * (1.0f - cosTheta); // Uniform distribution in solid angle
    float r = sqrt(max(0.0f, 1.0f - z * z));
    float phi = 2.0f * 3.14159265f * u2;
    
    float x = r * cos(phi);
    float y = r * sin(phi);
    
    // Create orthonormal basis around the primary direction
    float3 up = abs(dir.z) < 0.999f ? float3(0, 0, 1) : float3(1, 0, 0);
    float3 tangent = normalize(cross(up, dir));
    float3 bitangent = cross(dir, tangent);
    
    return tangent * x + bitangent * y + dir * z;
}

// Map uniform variables to a cosine-weighted direction for AO
float3 getCosineHemisphereSample(float u1, float u2, float3 normal)
{
    float r = sqrt(u1);
    float theta = 2.0f * 3.14159265f * u2;

    float x = r * cos(theta);
    float y = r * sin(theta);
    float z = sqrt(max(0.0f, 1.0f - u1));

    float3 up = abs(normal.z) < 0.999f ? float3(0, 0, 1) : float3(1, 0, 0);
    float3 tangent = normalize(cross(up, normal));
    float3 bitangent = cross(normal, tangent);

    return tangent * x + bitangent * y + normal * z;
}

// Fast RNG hashing
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

// Convert float4 to int4 and back to allow atomic adds in compute shaders
float4 unpackFloat4(int4 x)
{
    return float4(x) / FLOAT4_PACKING_CONSTANT;
}
int4 packFloat4(float4 x)
{
    return int4(x * FLOAT4_PACKING_CONSTANT);
}

uint get1DIndex(uint i, uint j, uint R)
{
    return i * (R + 1) - i * (i - 1) / 2 + j;
}

// Calculate subdivision grid weights for spatial interpolation
void getMeshColorIndicesAndWeights(float3 barycentrics, uint R,
                                   out uint i0, out uint j0, out float w0,
                                   out uint i1, out uint j1, out float w1,
                                   out uint i2, out uint j2, out float w2)
{
    barycentrics = saturate(barycentrics);
    barycentrics /= (barycentrics.x + barycentrics.y + barycentrics.z);

    float u_prime = barycentrics.x * R;
    float v_prime = barycentrics.y * R;

    int i = (int) floor(u_prime);
    if (i >= (int) R)
        i = R - 1;

    int j = (int) floor(v_prime);
    if (i + j >= (int) R)
        j = R - 1 - i;

    float du = u_prime - i;
    float dv = v_prime - j;

    if (du + dv <= 1.0f)
    {
        i0 = i;
        j0 = j;
        w0 = 1.0f - du - dv;
        i1 = i + 1;
        j1 = j;
        w1 = du;
        i2 = i;
        j2 = j + 1;
        w2 = dv;
    }
    else
    {
        i0 = i + 1;
        j0 = j + 1;
        w0 = du + dv - 1.0f;
        i1 = i + 1;
        j1 = j;
        w1 = 1.0f - dv;
        i2 = i;
        j2 = j + 1;
        w2 = 1.0f - du;
        
        if (i0 + j0 > R)
        {
            i0 = i;
            j0 = j;
        }
    }
}

// Vectorized activation functions
float4 activationFunction(float4 v)
{
    return max(v, v * LEAKY_RELU_SLOPE);
}

float4 activationFunctionDeriv(float4 v)
{
    return lerp((float4) LEAKY_RELU_SLOPE, (float4) 1.0f, step(0.0f, v));
}

float4 activationFunctionOutput(float4 v)
{
    return 1.0f / (1.0f + exp(-v));
}

float4 activationFunctionOutputDeriv(float4 v)
{
    return v * (1.0f - v);
}

// MLP forward pass layer execution
void evalLayer(inout float4 previousActivations[MAX_FEATURE_QUARTETS], inout float4 currentActivations[MAX_FEATURE_QUARTETS],
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
        currentActivations[neuronQuartet] = (layerType == HIDDEN_LAYER) ? activationFunction(neuronValue + bias) : activationFunctionOutput(neuronValue + bias);
    }
}

#ifndef GATE_INFERENCE

// Safely accumulate gradients across threads using atomic additions
void accumulateGradient(RWStructuredBuffer<int4> gradientTarget, const uint gradientIndex, float4 gradient)
{
    gradient = clamp(gradient, -0.5f, 0.5f);
    const int4 packed = packFloat4(gradient);
    InterlockedAdd(gradientTarget[gradientIndex].x, packed.x);
    InterlockedAdd(gradientTarget[gradientIndex].y, packed.y);
    InterlockedAdd(gradientTarget[gradientIndex].z, packed.z);
    InterlockedAdd(gradientTarget[gradientIndex].w, packed.w);
}

// Interpolate feature inputs for the active triangle
void gateEncoding(const GateEncodingData gateData, inout uint activationIndex, inout float4 activations[ACTIVATION_QUARTETS_PER_NETWORK])
{
    uint base0 = gateData.indices.x * featureQuartets;
    uint base1 = gateData.indices.y * featureQuartets;
    uint base2 = gateData.indices.z * featureQuartets;

    for (uint q = 0; q < featureQuartets; ++q)
    {
        float4 f0 = UniqueFeatureBuffer[base0 + q];
        float4 f1 = UniqueFeatureBuffer[base1 + q];
        float4 f2 = UniqueFeatureBuffer[base2 + q];
        
        float4 act = gateData.barycentrics.x * f0 + gateData.barycentrics.y * f1 + gateData.barycentrics.z * f2;

        // Mask off unallocated dimension channels
        if (q * 4 + 0 >= featureFloats)
            act.x = 0.0f;
        if (q * 4 + 1 >= featureFloats)
            act.y = 0.0f;
        if (q * 4 + 2 >= featureFloats)
            act.z = 0.0f;
        if (q * 4 + 3 >= featureFloats)
            act.w = 0.0f;

        activations[activationIndex++] = act;
    }
}

// Push gradient errors back into the spatial feature grid
void gateEncodingBackprop(const GateEncodingData gateData, inout float4 errors[ACTIVATION_QUARTETS_PER_NETWORK])
{
    for (uint q = 0; q < featureQuartets; ++q)
    {
        float4 inputGrad = errors[q];
        
        // Prevent unused features from drifting via stray gradients
        if (q * 4 + 0 >= featureFloats)
            inputGrad.x = 0.0f;
        if (q * 4 + 1 >= featureFloats)
            inputGrad.y = 0.0f;
        if (q * 4 + 2 >= featureFloats)
            inputGrad.z = 0.0f;
        if (q * 4 + 3 >= featureFloats)
            inputGrad.w = 0.0f;

        accumulateGradient(FeatureGradientBuffer, gateData.indices.x * featureQuartets + q, inputGrad * gateData.barycentrics.x);
        accumulateGradient(FeatureGradientBuffer, gateData.indices.y * featureQuartets + q, inputGrad * gateData.barycentrics.y);
        accumulateGradient(FeatureGradientBuffer, gateData.indices.z * featureQuartets + q, inputGrad * gateData.barycentrics.z);
    }
}

// Execute forward pass (training version)
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

// Calculate gradients and propagate error backwards through the layer
void backpropLayer(const float4 target, inout float4 activations[ACTIVATION_QUARTETS_PER_NETWORK], inout float4 errors[ACTIVATION_QUARTETS_PER_NETWORK], uint prevLayerQuartets, uint currLayerQuartets, uint prevOffset, uint currOffset, uint weightIndex, uint layerType)
{
    for (uint pq = prevOffset; pq < prevOffset + prevLayerQuartets; pq++)
        errors[pq] = 0.0f;

    for (uint q = currOffset; q < currOffset + currLayerQuartets; q++)
    {
        const float4 act = activations[q];
        float4 dCost_O = (layerType == OUTPUT_LAYER) ? (act - target) : errors[q];
        const float4 dCost_Z = dCost_O * ((layerType == HIDDEN_LAYER) ? activationFunctionDeriv(act) : activationFunctionOutputDeriv(act));
        
        for (uint prevQ = prevOffset; prevQ < prevOffset + prevLayerQuartets; prevQ++)
        {
            const float4 prevAct = activations[prevQ];
            float4 wX = MLPParameterBuffer[weightIndex];
            float4 wY = MLPParameterBuffer[weightIndex + 1];
            float4 wZ = MLPParameterBuffer[weightIndex + 2];
            float4 wW = MLPParameterBuffer[weightIndex + 3];

            errors[prevQ] += wX * dCost_Z.x + wY * dCost_Z.y + wZ * dCost_Z.z + wW * dCost_Z.w;

            accumulateGradient(MLPGradientBuffer, weightIndex++, dCost_Z.x * prevAct);
            accumulateGradient(MLPGradientBuffer, weightIndex++, dCost_Z.y * prevAct);
            accumulateGradient(MLPGradientBuffer, weightIndex++, dCost_Z.z * prevAct);
            accumulateGradient(MLPGradientBuffer, weightIndex++, dCost_Z.w * prevAct);
        }
        accumulateGradient(MLPGradientBuffer, weightIndex++, dCost_Z);
    }
}

// AdamW optimizer step calculation
float4 ApplyAdam(float4 gradient, float4 currentValue, inout AdamData adamData, float lr, float wd)
{
    adamData.stepCount = min(adamData.stepCount + 1, 1024u);
    
    float localBeta1T = pow(adamBeta1, (float) adamData.stepCount);
    float localBeta2T = pow(adamBeta2, (float) adamData.stepCount);

    adamData.mean = lerp(gradient, adamData.mean, adamBeta1);
    adamData.variance = lerp(gradient * gradient, adamData.variance, adamBeta2);
    
    float4 correctedMean = adamData.mean / (1.0f - localBeta1T);
    float4 correctedVariance = adamData.variance / (1.0f - localBeta2T);
    
    float4 adamStep = correctedMean * rsqrt(correctedVariance + adamEpsilon);
    float4 decayStep = currentValue * wd;

    return -lr * (adamStep + decayStep);
}
#endif // !GATE_INFERENCE