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

    // 1. Screen-Space Importance Sampling (The "16 Candidates" Loop)
    uint bestTriID = 0;
    uint lowestStepCount = 0xFFFFFFFF; // Start with max value
    bool foundValidCandidate = false;

    float strategyRoll = rand(rng); 

    if (strategyRoll < screenSpaceRatio) 
    {
        for (int i = 0; i < 16; ++i)
        {
            // Throw a random dart at the screen
            uint2 pixelCoord = uint2(
                (uint)(rand(rng) * screenWidth), 
                (uint)(rand(rng) * screenHeight)
            );

            // Look up the Triangle ID visible at this pixel
            uint candidateTriID = VisibilityBuffer.Load(int3(pixelCoord, 0)).r;

            // If it hit the skybox (often represented as 0xFFFFFFFF), skip it
            if (candidateTriID >= totalTriangles)
                continue;

            // Fetch the global triangle to find its vertices
            GlobalTriangle candidateTri = TriangleBuffer[candidateTriID];

            // Read the Adam step counts for these 3 vertices (multiplying by 2 because 
            // you have 2 float4s per vertex in your GateFeature struct)
            uint step0 = GateFeatureAdamBuffer[candidateTri.i0 * 2].stepCount;
            uint step1 = GateFeatureAdamBuffer[candidateTri.i1 * 2].stepCount;
            uint step2 = GateFeatureAdamBuffer[candidateTri.i2 * 2].stepCount;

            // Calculate how "trained" this triangle is
            uint avgStep = (step0 + step1 + step2) / 3;

            // If this is the most undertrained triangle we've seen so far, save it
            if (avgStep < lowestStepCount)
            {
                lowestStepCount = avgStep;
                bestTriID = candidateTriID;
                foundValidCandidate = true;
            }
        }
    }
    else 
    {
        bestTriID = min((uint)(rand(rng) * totalTriangles), totalTriangles - 1);
    }
    //uint bestTriID = min((uint)(rand(rng) * totalTriangles), totalTriangles - 1);

    // Now we have our absolute best candidate! Proceed as normal.
    GlobalTriangle tri = TriangleBuffer[bestTriID];
    uint i0 = tri.i0;
    uint i1 = tri.i1;
    uint i2 = tri.i2;

    // 2. Uniform Barycentrics
    float u1 = rand(rng);
    float u2 = rand(rng);
    float sqrt_u1 = sqrt(u1);
    GateEncodingData gateData;
    gateData.barycentrics = float3(1.0f - sqrt_u1, sqrt_u1 * (1.0f - u2), sqrt_u1 * u2);
    gateData.indices = uint3(i0, i1, i2);

    // 3. Ground Truth Sampling
    float2 uv0 = asfloat(VertexUVBuffer.Load2(i0 * VertexStride + uvOffset));
    float2 uv1 = asfloat(VertexUVBuffer.Load2(i1 * VertexStride + uvOffset));
    float2 uv2 = asfloat(VertexUVBuffer.Load2(i2 * VertexStride + uvOffset));
    float2 interpUV = gateData.barycentrics.x * uv0 + gateData.barycentrics.y * uv1 + gateData.barycentrics.z * uv2;
    float3 target = BindlessTextures[tri.materialIdx * 6].SampleLevel(LinearSampler, interpUV, 0).rgb;

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