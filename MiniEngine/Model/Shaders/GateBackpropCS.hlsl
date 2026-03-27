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

            // If it hit the skybox, skip it
            if (candidateTriID >= totalTriangles)
                continue;

            // OPRAVA: Zde musíme použít SpatialTriangleBuffer, abychom nezajeli mimo Adam Buffer!
            GlobalTriangle candidateSpatialTri = SpatialTriangleBuffer[candidateTriID];

            // Read the Adam step counts using the UNIQUE SPATIAL indices
            uint step0 = GateFeatureAdamBuffer[candidateSpatialTri.i0 * 2].stepCount;
            uint step1 = GateFeatureAdamBuffer[candidateSpatialTri.i1 * 2].stepCount;
            uint step2 = GateFeatureAdamBuffer[candidateSpatialTri.i2 * 2].stepCount;

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

    // OPRAVA: Pokud jsme vybrali world-space strategii, NEBO pokud screen-space 
    // nenašel žádný platný cíl (koukáme do prázdna), vybereme náhodný trojúhelník.
    if (!foundValidCandidate) 
    {
        bestTriID = min((uint)(rand(rng) * totalTriangles), totalTriangles - 1);
    }

    // 1. Získej OBA trojúhelníky
    GlobalTriangle origTri = TriangleBuffer[bestTriID];
    GlobalTriangle spatialTri = SpatialTriangleBuffer[bestTriID];

    // 2. Uniform Barycentrics
    float u1 = rand(rng);
    float u2 = rand(rng);
    float sqrt_u1 = sqrt(u1);
    GateEncodingData gateData;
    gateData.barycentrics = float3(1.0f - sqrt_u1, sqrt_u1 * (1.0f - u2), sqrt_u1 * u2);
    
    // Pro backprop a MLP posíláme spatial indexy (unikátní vertexy)
    gateData.indices = uint3(spatialTri.i0, spatialTri.i1, spatialTri.i2);

    // 3. Ground Truth Sampling (Pro UV textury používáme pùvodní indexy)
    float2 uv0 = asfloat(VertexUVBuffer.Load2(origTri.i0 * VertexStride + uvOffset));
    float2 uv1 = asfloat(VertexUVBuffer.Load2(origTri.i1 * VertexStride + uvOffset));
    float2 uv2 = asfloat(VertexUVBuffer.Load2(origTri.i2 * VertexStride + uvOffset));
    float2 interpUV = gateData.barycentrics.x * uv0 + gateData.barycentrics.y * uv1 + gateData.barycentrics.z * uv2;
    float3 target = BindlessTextures[origTri.materialIdx * 6].SampleLevel(LinearSampler, interpUV, 0).rgb;

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