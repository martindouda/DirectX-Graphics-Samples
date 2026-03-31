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

            uint baseIdx = candidateTriID * pointsPerTri;
            
            uint randomPt = min((uint)(rand(rng) * pointsPerTri), pointsPerTri - 1);
            
            // Získáme unikátní ID pro tento jeden bod
            uint uniquePt = VertexMappingBuffer[baseIdx + randomPt]; 

            // Pøeèteme poèet krokù z Adam bufferu POUZE pro tento vybraný bod
            uint stepCount = FeatureAdamBuffer[uniquePt * 2].stepCount;

            // Zkontrolujeme, jestli je tento bod ménì natrénovaný než náš dosavadní nejhorší
            if (stepCount < lowestStepCount)
            {
                lowestStepCount = stepCount;
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

    // Získáme lokální møížkové souøadnice
    uint i0, j0, i1, j1, i2, j2;
    float weight0, weight1, weight2;
    getMeshColorIndicesAndWeights(barycentrics, meshColorResolution, i0, j0, weight0, i1, j1, weight1, i2, j2, weight2);

    uint idx0 = get1DIndex(i0, j0, meshColorResolution);
    uint idx1 = get1DIndex(i1, j1, meshColorResolution);
    uint idx2 = get1DIndex(i2, j2, meshColorResolution);

    uint baseIndex = bestTriID * pointsPerTri;
    GateEncodingData gateData;
    gateData.barycentrics = float3(weight0, weight1, weight2);
    
    // Pøevod z duplikovaného indexu møížky na unikátní ID!
    gateData.indices.x = VertexMappingBuffer[baseIndex + idx0];
    gateData.indices.y = VertexMappingBuffer[baseIndex + idx1];
    gateData.indices.z = VertexMappingBuffer[baseIndex + idx2];

    // For ground truth we use the original triangle to sample the texture, because that's where the UVs are
    GlobalTriangle origTri = GlobalTriangleBuffer[bestTriID];
    float2 uv0 = asfloat(VertexUVBuffer.Load2(origTri.i0 * VertexStride + uvOffset));
    float2 uv1 = asfloat(VertexUVBuffer.Load2(origTri.i1 * VertexStride + uvOffset));
    float2 uv2 = asfloat(VertexUVBuffer.Load2(origTri.i2 * VertexStride + uvOffset));
    // Použijeme pùvodní globální 'barycentrics', nikoliv lokální 'gateData.barycentrics'!
    float2 interpUV = barycentrics.x * uv0 + barycentrics.y * uv1 + barycentrics.z * uv2;
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

    float3 diff = target - activations[6].xyz; // Output vrstva zaèíná na indexu 36
    float pixelLoss = dot(diff, diff); // MSE (Mean Squared Error)
    
    // Trik: vynásobíme milionem a bezpeènì seèteme ze všech vláken
    LossBuffer.InterlockedAdd(0, (uint)(pixelLoss * 1000.0f));
}