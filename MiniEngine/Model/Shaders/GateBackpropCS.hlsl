// File: GateBackpropCS.hlsl

#include "GateTrainCommon.hlsli"


// =========================================================================
//   KERNEL 1: Forward Pass & Gradient Accumulation
// =========================================================================

[numthreads(BACKPROP_THREADGROUP_SIZE, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint linearIndex = DTid.x;
    uint rng = pcgHash(linearIndex ^ pcgHash(trainingStep));

    // 1. Uniform Triangle Sampling
    uint triID = min((uint)(rand(rng) * totalTriangles), totalTriangles - 1);
    
    // Assumes 16-bit index buffer (standard for many Sponza meshes, adjust if 32-bit!)
    uint indexOffset = triID * 3 * 2;
    uint dword0 = IndexBuffer.Load(indexOffset & ~3);
    uint dword1 = IndexBuffer.Load((indexOffset + 4) & ~3);
    
    uint i0 = (indexOffset % 4 == 0) ? (dword0 & 0xFFFF) : (dword0 >> 16);
    uint i1 = (indexOffset % 4 == 0) ? (dword0 >> 16) : (dword1 & 0xFFFF);
    uint i2 = (indexOffset % 4 == 0) ? (dword1 & 0xFFFF) : (dword1 >> 16);

    // 2. Uniform Barycentrics
    float u1 = rand(rng);
    float u2 = rand(rng);
    float sqrt_u1 = sqrt(u1);
    GateEncodingData gateData;
    gateData.barycentrics = float3(1.0f - sqrt_u1, sqrt_u1 * (1.0f - u2), sqrt_u1 * u2);
    gateData.indices = uint3(i0, i1, i2);

    // 3. Ground Truth Sampling
    float2 uv0 = asfloat(VertexUVBuffer.Load2(i0 * 8)); // Assumes 8-byte (float2) UVs
    float2 uv1 = asfloat(VertexUVBuffer.Load2(i1 * 8));
    float2 uv2 = asfloat(VertexUVBuffer.Load2(i2 * 8));
    float2 interpUV = gateData.barycentrics.x * uv0 + gateData.barycentrics.y * uv1 + gateData.barycentrics.z * uv2;
    float3 target = GroundTruthTexture.SampleLevel(LinearSampler, interpUV, 0).rgb;

    // 4. FORWARD PASS
    float4 activations[ACTIVATION_QUARTETS_PER_NETWORK];
    uint activationIndex = 0;
    
    gateEncoding(gateData, activationIndex, activations); // Layer 0 (Input)
    evalLayerActivations(activations, 0,  0, 2, 4, 2, HIDDEN_LAYER); // Layer 1 (Hidden)
    evalLayerActivations(activations, 36, 2, 6, 1, 4, OUTPUT_LAYER); // Layer 2 (Output)

    // 5. BACKWARD PASS
    float4 errors[ACTIVATION_QUARTETS_PER_NETWORK];
    backpropLayer(target, activations, errors, 4, 1, 2, 6, 36, OUTPUT_LAYER); // Output -> Hidden
    backpropLayer(target, activations, errors, 2, 4, 0, 2, 0,  HIDDEN_LAYER); // Hidden -> Input
    gateEncodingBackprop(gateData, errors); // Distribute to Vertices
}