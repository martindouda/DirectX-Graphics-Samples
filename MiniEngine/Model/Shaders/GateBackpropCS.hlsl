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

    uint bestTriID = 0;
    uint lowestStepCount = 0xFFFFFFFF;
    bool foundValidCandidate = false;

    float strategyRoll = rand(rng); 

    if (strategyRoll < screenSpaceRatio) 
    {
        for (int i = 0; i < 16; ++i)
        {
            uint2 pixelCoord = uint2((uint)(rand(rng) * screenWidth),  (uint)(rand(rng) * screenHeight));

            // Look up the Triangle ID visible at this pixel
            uint candidateTriID = VisibilityBuffer.Load(int3(pixelCoord, 0)).r;

            // If it hit the skybox, skip it
            if (candidateTriID >= totalTriangles)
                continue;

            // NAHRAZENO MÍSTO SpatialTriangleBuffer:
            uint baseIdx = candidateTriID * 6;
            
            // Získáme unikátní ID našich 3 rohù trojúhelníku pøímo z N:M mapování
            uint uniqueI0 = VertexMappingBuffer[baseIdx + 0]; // Roh U
            uint uniqueI1 = VertexMappingBuffer[baseIdx + 1]; // Roh V
            uint uniqueI2 = VertexMappingBuffer[baseIdx + 2]; // Roh W

            // Pøeèteme poèty krokù z Adam bufferu
            uint step0 = FeatureAdamBuffer[uniqueI0 * 2].stepCount;
            uint step1 = FeatureAdamBuffer[uniqueI1 * 2].stepCount;
            uint step2 = FeatureAdamBuffer[uniqueI2 * 2].stepCount;

            // Spoèítáme prùmìr natrénovanosti tohoto trojúhelníku
            uint avgStep = (step0 + step1 + step2) / 3;

            if (avgStep < lowestStepCount)
            {
                lowestStepCount = avgStep;
                bestTriID = candidateTriID;
                foundValidCandidate = true;
            }
        }

        if (!foundValidCandidate) 
            bestTriID = min((uint)(rand(rng) * totalTriangles), totalTriangles - 1);
    }
    else
    {
        bestTriID = min((uint)(rand(rng) * totalTriangles), totalTriangles - 1);
    }


    float u1 = rand(rng); float u2 = rand(rng); float sqrt_u1 = sqrt(u1);
    float3 barycentrics = float3(1.0f - sqrt_u1, sqrt_u1 * (1.0f - u2), sqrt_u1 * u2);

    // Mesh Colors R=2 logika
    float u = barycentrics.x * 2.0f; float v = barycentrics.y * 2.0f; float w = barycentrics.z * 2.0f;
    uint i0, i1, i2; float weight0, weight1, weight2;

    if (u >= 1.0f) { i0 = 0; i1 = 3; i2 = 5; weight0 = u - 1.0f; weight1 = v; weight2 = w; }
    else if (v >= 1.0f) { i0 = 3; i1 = 1; i2 = 4; weight0 = u; weight1 = v - 1.0f; weight2 = w; }
    else if (w >= 1.0f) { i0 = 5; i1 = 4; i2 = 2; weight0 = u; weight1 = v; weight2 = w - 1.0f; }
    else { i0 = 3; i1 = 4; i2 = 5; weight0 = 1.0f - w; weight1 = 1.0f - u; weight2 = 1.0f - v; }

    uint baseIndex = bestTriID * 6;
    GateEncodingData gateData;
    gateData.barycentrics = float3(weight0, weight1, weight2);
    
    // Klíèový krok: Pøevod z duplikovaného indexu møížky na unikátní ID!
    gateData.indices.x = VertexMappingBuffer[baseIndex + i0];
    gateData.indices.y = VertexMappingBuffer[baseIndex + i1];
    gateData.indices.z = VertexMappingBuffer[baseIndex + i2];

    // For ground truth we use the original triangle to sample the texture, because that's where the UVs are
    GlobalTriangle origTri = GlobalTriangleBuffer[bestTriID];
    float2 uv0 = asfloat(VertexUVBuffer.Load2(origTri.i0 * VertexStride + uvOffset));
    float2 uv1 = asfloat(VertexUVBuffer.Load2(origTri.i1 * VertexStride + uvOffset));
    float2 uv2 = asfloat(VertexUVBuffer.Load2(origTri.i2 * VertexStride + uvOffset));
    float2 interpUV = gateData.barycentrics.x * uv0 + gateData.barycentrics.y * uv1 + gateData.barycentrics.z * uv2;
    float3 target = BindlessTextures[origTri.materialIdx * 6].SampleLevel(LinearSampler, interpUV, 0).rgb;

    // Forward pass
    float4 activations[ACTIVATION_QUARTETS_PER_NETWORK];
    uint activationIndex = 0;
    
    gateEncoding(gateData, activationIndex, activations);               // Layer 0 (Input)
    evalLayerActivations(activations, 0,  0, 2, 4, 2, HIDDEN_LAYER);    // Layer 1 (Hidden)
    evalLayerActivations(activations, 36, 2, 6, 1, 4, OUTPUT_LAYER);    // Layer 2 (Output)

    // Backward pass
    float4 errors[ACTIVATION_QUARTETS_PER_NETWORK];
    backpropLayer(target, activations, errors, 4, 1, 2, 6, 36, OUTPUT_LAYER);   // Output -> Hidden
    backpropLayer(target, activations, errors, 2, 4, 0, 2, 0,  HIDDEN_LAYER);   // Hidden -> Input
    gateEncodingBackprop(gateData, errors);                                     // Distribute to Vertices
}