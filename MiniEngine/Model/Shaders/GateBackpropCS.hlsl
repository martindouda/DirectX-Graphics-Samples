// File: GateBackpropCS.hlsl

#include "GateTrainCommon.hlsli"

// =========================================================================
//  KERNEL: Forward Pass & Gradient Accumulation
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

    // --- Active Active Learning Strategy (Screen-Space vs Random) ---
    if (strategyRoll < screenSpaceRatio) 
    {
        for (int i = 0; i < 16; ++i)
        {
            uint2 pixelCoord = uint2((uint)(rand(rng) * screenWidth), (uint)(rand(rng) * screenHeight));
            uint rawID = VisibilityBuffer.Load(int3(pixelCoord, 0)).r;

            if (rawID == 0) // Zero is sky
                continue;

            uint candidateTriID = rawID - 1;

            if (candidateTriID >= totalTriangles)
                continue;

            uint baseIdx = candidateTriID * pointsPerTri;
            
            // Get a random point within this triangle
            uint randomPt = min((uint)(rand(rng) * pointsPerTri), pointsPerTri - 1);
            uint uniquePt = VertexMappingBuffer[baseIdx + randomPt]; 

            // Read the step count from the Adam buffer ONLY for this selected point
            uint stepCount = FeatureAdamBuffer[uniquePt * 2].stepCount;

            // Check if this point is less trained than our worst candidate so far
            if (stepCount < lowestStepCount)
            {
                lowestStepCount = stepCount;
                bestTriID = candidateTriID;
                foundValidCandidate = true;
            }
        }

        // Fallback if no valid visible triangle was found in the random samples
        if (!foundValidCandidate) 
            bestTriID = min((uint)(rand(rng) * totalTriangles), totalTriangles - 1);
    }
    else
    {
        // Random exploration
        bestTriID = min((uint)(rand(rng) * totalTriangles), totalTriangles - 1);
    }

    // --- Generate Random Barycentric Coordinates ---
    float u1 = rand(rng); 
    float u2 = rand(rng); 
    float sqrt_u1 = sqrt(u1);
    float3 barycentrics = float3(1.0f - sqrt_u1, sqrt_u1 * (1.0f - u2), sqrt_u1 * u2);

    // Get local grid coordinates
    uint i0, j0, i1, j1, i2, j2;
    float weight0, weight1, weight2;
    getMeshColorIndicesAndWeights(barycentrics, meshColorResolution, i0, j0, weight0, i1, j1, weight1, i2, j2, weight2);

    uint idx0 = get1DIndex(i0, j0, meshColorResolution);
    uint idx1 = get1DIndex(i1, j1, meshColorResolution);
    uint idx2 = get1DIndex(i2, j2, meshColorResolution);

    uint baseIndex = bestTriID * pointsPerTri;
    GateEncodingData gateData;
    gateData.barycentrics = float3(weight0, weight1, weight2);
    
    // Convert duplicated grid index to unique ID!
    gateData.indices.x = VertexMappingBuffer[baseIndex + idx0];
    gateData.indices.y = VertexMappingBuffer[baseIndex + idx1];
    gateData.indices.z = VertexMappingBuffer[baseIndex + idx2];

    GlobalTriangle origTri = GlobalTriangleBuffer[bestTriID];
    
    // --- 1. GET POSITION AND NORMAL FOR RAY QUERY ---
    // Assume position is at offset 0 (DXGI_FORMAT_R32G32B32_FLOAT) in the vertex buffer
    float3 p0 = asfloat(VertexUVBuffer.Load3(origTri.i0 * VertexStride));
    float3 p1 = asfloat(VertexUVBuffer.Load3(origTri.i1 * VertexStride));
    float3 p2 = asfloat(VertexUVBuffer.Load3(origTri.i2 * VertexStride));

    // World position of the sample
    float3 worldPos = barycentrics.x * p0 + barycentrics.y * p1 + barycentrics.z * p2;

    // Fast calculation of the geometric normal for offset (prevents self-shadowing)
    float3 faceNormal = normalize(cross(p1 - p0, p2 - p0));

    // --- 2. INLINE RAY TRACING (SHADOW QUERY) ---
    RayDesc ray;
    ray.Origin = worldPos + faceNormal * 0.05f; // Small offset along the normal prevents intersecting the source triangle
    ray.Direction = sunDirection;               // Must point TOWARDS the sun
    ray.TMin = 0.0f;
    ray.TMax = 10000.0f;

    // Flags optimized purely for shadows - we ONLY care if an opaque object blocks the ray
    RayQuery<RAY_FLAG_CULL_NON_OPAQUE | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
    q.TraceRayInline(SceneBVH, 0, 0xFF, ray);
    q.Proceed();

    bool isShadowed = (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT);

    // --- 3. GROUND TRUTH TARGET WITH SHADOW APPLICATION ---
    float target = isShadowed ? 0.f : 1.f;
    float4 targetInput = float4(target, 0.f, 0.f, 0.f);

    // --- 4. NETWORK FORWARD PASS ---
    float4 activations[ACTIVATION_QUARTETS_PER_NETWORK];
    uint activationIndex = 0;
    
    gateEncoding(gateData, activationIndex, activations);                // Layer 0 (Input)
    evalLayerActivations(activations, 0,  0, 2, 4, 2, HIDDEN_LAYER);     // Layer 1 (Hidden)
    evalLayerActivations(activations, 36, 2, 6, 1, 4, OUTPUT_LAYER);     // Layer 2 (Output)

    // --- 5. NETWORK BACKWARD PASS ---
    float4 errors[ACTIVATION_QUARTETS_PER_NETWORK];
    backpropLayer(targetInput, activations, errors, 4, 1, 2, 6, 36, OUTPUT_LAYER);   // Output -> Hidden
    backpropLayer(targetInput, activations, errors, 2, 4, 0, 2, 0,  HIDDEN_LAYER);   // Hidden -> Input
    gateEncodingBackprop(gateData, errors);                                          // Distribute to Vertices

    // --- 6. LOSS ACCUMULATION ---
    float diff = target - activations[6].x; // Output layer starts at index 36 (36 / 4 = 9, assuming tight packing, but using activation index mapped logically)
    float pixelLoss = dot(diff, diff);      // MSE
    
    LossBuffer.InterlockedAdd(0, (uint)(pixelLoss * 1000.0f));
}