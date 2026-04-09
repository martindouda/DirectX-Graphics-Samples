// File: GateBackpropCS.hlsl

#include "GateTrainCommon.hlsli"

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

[numthreads(BACKPROP_THREADGROUP_SIZE, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint linearIndex = DTid.x;
    uint rng = pcgHash(linearIndex ^ pcgHash(trainingStep));

    uint bestTriID = 0;
    uint lowestStepCount = 0xFFFFFFFF;
    bool foundValidCandidate = false;

    float strategyRoll = rand(rng); 

    // --- Active Learning Strategy ---
    // Prioritizes training on pixels that are visible on-screen and have the 
    // lowest Adam step count (meaning they have been trained the least).
    if (strategyRoll < screenSpaceRatio) 
    {
        for (int i = 0; i < 16; ++i)
        {
            uint2 pixelCoord = uint2((uint)(rand(rng) * screenWidth), (uint)(rand(rng) * screenHeight));
            uint rawID = VisibilityBuffer.Load(int3(pixelCoord, 0)).r;

            if (rawID == 0) continue; // Skip Sky

            uint candidateTriID = rawID - 1;

            if (candidateTriID >= totalTriangles) continue;

            GlobalTriangle candidateTri = GlobalTriangleBuffer[candidateTriID];
            uint baseIdx = candidateTri.pointOffset;
            uint localPts = candidateTri.pointsPerTri;
            
            uint randomPt = min((uint)(rand(rng) * localPts), localPts - 1);
            uint uniquePt = VertexMappingBuffer[baseIdx + randomPt];

            // Fetch the Adam step count from the first quartet of the vertex to determine age
            uint stepCount = FeatureAdamBuffer[uniquePt * featureQuartets].stepCount;

            if (stepCount < lowestStepCount)
            {
                lowestStepCount = stepCount;
                bestTriID = candidateTriID;
                foundValidCandidate = true;
            }
        }

        // Fallback to purely random exploration if no valid screen triangle was found
        if (!foundValidCandidate) 
            bestTriID = min((uint)(rand(rng) * totalTriangles), totalTriangles - 1);
    }
    else
    {
        bestTriID = min((uint)(rand(rng) * totalTriangles), totalTriangles - 1);
    }

    float u1 = rand(rng); 
    float u2 = rand(rng); 
    float sqrt_u1 = sqrt(u1);
    float3 barycentrics = float3(1.0f - sqrt_u1, sqrt_u1 * (1.0f - u2), sqrt_u1 * u2);

    GlobalTriangle origTri = GlobalTriangleBuffer[bestTriID];
    uint baseIndex = origTri.pointOffset;
    uint localRes = origTri.resolution;

    uint i0, j0, i1, j1, i2, j2;
    float weight0, weight1, weight2;
    getMeshColorIndicesAndWeights(barycentrics, localRes, i0, j0, weight0, i1, j1, weight1, i2, j2, weight2);

    uint idx0 = get1DIndex(i0, j0, localRes);
    uint idx1 = get1DIndex(i1, j1, localRes);
    uint idx2 = get1DIndex(i2, j2, localRes);

    GateEncodingData gateData;
    gateData.barycentrics = float3(weight0, weight1, weight2);
    gateData.indices.x = VertexMappingBuffer[baseIndex + idx0];
    gateData.indices.y = VertexMappingBuffer[baseIndex + idx1];
    gateData.indices.z = VertexMappingBuffer[baseIndex + idx2];
    
    float3 p0 = asfloat(VertexUVBuffer.Load3(origTri.i0 * VertexStride));
    float3 p1 = asfloat(VertexUVBuffer.Load3(origTri.i1 * VertexStride));
    float3 p2 = asfloat(VertexUVBuffer.Load3(origTri.i2 * VertexStride));

    float3 worldPos = barycentrics.x * p0 + barycentrics.y * p1 + barycentrics.z * p2;

    // 1. Load the Vertex Normals (Offset by 20 bytes)
    float3 n0 = asfloat(VertexUVBuffer.Load3(origTri.i0 * VertexStride + 20));
    float3 n1 = asfloat(VertexUVBuffer.Load3(origTri.i1 * VertexStride + 20));
    float3 n2 = asfloat(VertexUVBuffer.Load3(origTri.i2 * VertexStride + 20));

    // 2. Interpolate them using the barycentric weights to get perfectly smooth curves
    float3 smoothNormal = normalize(barycentrics.x * n0 + barycentrics.y * n1 + barycentrics.z * n2);

    // --- DXR Inline Ray Tracing (Ground Truth Generation) ---
    // 1. Trace a directional ray towards the sun for hard shadows
    RayDesc shadowRay;
    shadowRay.Origin = worldPos + smoothNormal * 0.05f;
    shadowRay.Direction = sunDirection;
    shadowRay.TMin = 0.0f;
    shadowRay.TMax = 10000.0f;

    RayQuery<RAY_FLAG_CULL_NON_OPAQUE | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> qShadow;
    qShadow.TraceRayInline(SceneBVH, 0, 0xFF, shadowRay);
    qShadow.Proceed();
    bool isShadowed = (qShadow.CommittedStatus() == COMMITTED_TRIANGLE_HIT);

    // 2. Trace a random cosine-weighted ray for Ambient Occlusion
    float u3 = rand(rng);
    float u4 = rand(rng);
    float3 aoDirection = getCosineHemisphereSample(u3, u4, smoothNormal);

    RayDesc aoRay;
    aoRay.Origin = worldPos + smoothNormal * 0.05f;
    aoRay.Direction = aoDirection;                
    aoRay.TMin = 0.0f;
    aoRay.TMax = aoRadius;                

    RayQuery<RAY_FLAG_CULL_NON_OPAQUE | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> qAO;
    qAO.TraceRayInline(SceneBVH, 0, 0xFF, aoRay);
    qAO.Proceed();
    bool isOccluded = (qAO.CommittedStatus() == COMMITTED_TRIANGLE_HIT);

    // Target format: (Shadow, AO, unused, unused)
    float targetShadow = isShadowed ? 0.0f : 1.0f;
    float targetAO = isOccluded ? 0.0f : 1.0f;
    float4 targetInput = float4(targetShadow, targetAO, 0.0f, 0.0f);

    // --- Dynamic Network Mathematics ---
    // The MLP buffer is a flat array of weights. We must calculate the offset for the output layer
    // based on the dynamic feature size.
    // Hidden Layer = 16 neurons. Output Layer = 4 neurons.
    // Weight parameters for Hidden Layer = 16 neurons * featureQuartets inputs + 16 biases
    // Since we process in quartets (groups of 4 floats), the offset is (16/4)*featureQuartets*4 + 16/4
    // Which simplifies to: 16 * featureQuartets + 4
    uint hiddenOffset = 0;
    uint outputOffset = (16 * featureQuartets) + 4;

    float4 activations[ACTIVATION_QUARTETS_PER_NETWORK];
    uint activationIndex = 0;
    
    // Evaluate Forward Pass
    gateEncoding(gateData, activationIndex, activations);                
    evalLayerActivations(activations, hiddenOffset, 0,               featureQuartets,     4, featureQuartets, HIDDEN_LAYER);      
    evalLayerActivations(activations, outputOffset, featureQuartets, featureQuartets + 4, 1, 4,               OUTPUT_LAYER);      

    // Evaluate Backward Pass
    float4 errors[ACTIVATION_QUARTETS_PER_NETWORK];
    backpropLayer(targetInput, activations, errors, 4,               1, featureQuartets, featureQuartets + 4, outputOffset, OUTPUT_LAYER);   
    backpropLayer(targetInput, activations, errors, featureQuartets, 4, 0,               featureQuartets,     hiddenOffset, HIDDEN_LAYER);   
    gateEncodingBackprop(gateData, errors);                                          

    // Loss Accumulation for GUI visualization
    float2 diff = targetInput.xy - activations[featureQuartets + 4].xy; 
    float pixelLoss = dot(diff, diff) * 0.5f; 
    
    LossBuffer.InterlockedAdd(0, (uint)(pixelLoss * 1000.0f));
}