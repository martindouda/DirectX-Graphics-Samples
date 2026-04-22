#include "GateTrainCommon.hlsli"

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

[numthreads(BACKPROP_THREADGROUP_SIZE, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint linearIndex = DTid.x;
    uint rng = pcgHash(linearIndex ^ pcgHash(trainingStep));

    uint bestTriID = 0;
    uint lowestStepCount = 0xFFFFFFFF;
    bool foundValidCandidate = false;

    // Prioritize visible pixels with low Adam step counts
    if (rand(rng) < screenSpaceRatio) 
    {
        for (int i = 0; i < 16; ++i)
        {
            uint2 pixelCoord = uint2((uint)(rand(rng) * screenWidth), (uint)(rand(rng) * screenHeight));
            uint rawID = VisibilityBuffer.Load(int3(pixelCoord, 0)).r;

            if (rawID == 0) continue; 

            uint candidateTriID = rawID - 1;
            if (candidateTriID >= totalTriangles) continue;

            GlobalTriangle candidateTri = GlobalTriangleBuffer[candidateTriID];
            uint randomPt = min((uint)(rand(rng) * candidateTri.pointsPerTri), candidateTri.pointsPerTri - 1);
            
            // Check feature age via Adam step count
            uint uniquePt = VertexMappingBuffer[candidateTri.pointOffset + randomPt];
            uint stepCount = FeatureAdamBuffer[uniquePt * featureQuartets].stepCount;

            if (stepCount < lowestStepCount)
            {
                lowestStepCount = stepCount;
                bestTriID = candidateTriID;
                foundValidCandidate = true;
            }
        }
    }

    // Fallback to random uniform selection
    if (!foundValidCandidate) 
        bestTriID = min((uint)(rand(rng) * totalTriangles), totalTriangles - 1);

    // Uniform triangle sampling
    float u1 = rand(rng); 
    float u2 = rand(rng); 
    float sqrt_u1 = sqrt(u1);
    float3 barycentrics = float3(1.0f - sqrt_u1, sqrt_u1 * (1.0f - u2), sqrt_u1 * u2);

    GlobalTriangle origTri = GlobalTriangleBuffer[bestTriID];
    uint baseIndex = origTri.pointOffset;
    uint localRes = origTri.resolution;

    // Resolve grid indices for interpolation
    uint i0, j0, i1, j1, i2, j2;
    float weight0, weight1, weight2;
    getMeshColorIndicesAndWeights(barycentrics, localRes, i0, j0, weight0, i1, j1, weight1, i2, j2, weight2);

    GateEncodingData gateData;
    gateData.barycentrics = float3(weight0, weight1, weight2);
    gateData.indices.x = VertexMappingBuffer[baseIndex + get1DIndex(i0, j0, localRes)];
    gateData.indices.y = VertexMappingBuffer[baseIndex + get1DIndex(i1, j1, localRes)];
    gateData.indices.z = VertexMappingBuffer[baseIndex + get1DIndex(i2, j2, localRes)];
    
    // Evaluate DXR ground truth (Shadows and AO)
    float3 p0 = asfloat(VertexUVBuffer.Load3(origTri.i0 * VertexStride));
    float3 p1 = asfloat(VertexUVBuffer.Load3(origTri.i1 * VertexStride));
    float3 p2 = asfloat(VertexUVBuffer.Load3(origTri.i2 * VertexStride));
    float3 worldPos = barycentrics.x * p0 + barycentrics.y * p1 + barycentrics.z * p2;

    float3 n0 = asfloat(VertexUVBuffer.Load3(origTri.i0 * VertexStride + 20));
    float3 n1 = asfloat(VertexUVBuffer.Load3(origTri.i1 * VertexStride + 20));
    float3 n2 = asfloat(VertexUVBuffer.Load3(origTri.i2 * VertexStride + 20));
    float3 smoothNormal = normalize(barycentrics.x * n0 + barycentrics.y * n1 + barycentrics.z * n2);

    // Trace shadow ray
    RayDesc shadowRay = { worldPos + smoothNormal * 0.05f, 0.0f, sunDirection, 10000.0f };
    RayQuery<RAY_FLAG_CULL_NON_OPAQUE | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> qShadow;
    qShadow.TraceRayInline(SceneBVH, 0, 0xFF, shadowRay);
    qShadow.Proceed();
    bool isShadowed = (qShadow.CommittedStatus() == COMMITTED_TRIANGLE_HIT);

    // Trace AO ray
    RayDesc aoRay = { worldPos + smoothNormal * 0.05f, 0.0f, getCosineHemisphereSample(rand(rng), rand(rng), smoothNormal), aoRadius };
    RayQuery<RAY_FLAG_CULL_NON_OPAQUE | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> qAO;
    qAO.TraceRayInline(SceneBVH, 0, 0xFF, aoRay);
    qAO.Proceed();
    bool isOccluded = (qAO.CommittedStatus() == COMMITTED_TRIANGLE_HIT);

    // Execute MLP forward pass
    uint hiddenOffset = 0;
    uint outputOffset = (16 * featureQuartets) + 4;

    float4 activations[ACTIVATION_QUARTETS_PER_NETWORK];
    uint activationIndex = 0;
    
    gateEncoding(gateData, activationIndex, activations);                
    evalLayerActivations(activations, hiddenOffset, 0, featureQuartets, 4, featureQuartets, HIDDEN_LAYER);      
    evalLayerActivations(activations, outputOffset, featureQuartets, featureQuartets + 4, 1, 4, OUTPUT_LAYER);      

    // Determine target values based on learning mode
    float4 targetInput = float4(isShadowed ? 0.0f : 1.0f, isOccluded ? 0.0f : 1.0f, 0.0f, 0.0f);
    
    if (learningMode == 1) targetInput.x = activations[featureQuartets + 4].x;
    else if (learningMode == 2) targetInput.y = activations[featureQuartets + 4].y;
    else if (learningMode == 3)
    {
        // Calculate true surface radiance for RGB target
        float2 uv0 = asfloat(VertexUVBuffer.Load2(origTri.i0 * VertexStride + uvOffset));
        float2 uv1 = asfloat(VertexUVBuffer.Load2(origTri.i1 * VertexStride + uvOffset));
        float2 uv2 = asfloat(VertexUVBuffer.Load2(origTri.i2 * VertexStride + uvOffset));
        
        float2 pixelUV = barycentrics.x * uv0 + barycentrics.y * uv1 + barycentrics.z * uv2;
        float3 albedo = BindlessTextures[origTri.materialIdx * 6 + 0].SampleLevel(LinearSampler, pixelUV, 0).rgb;
        
        float3 directLight = albedo * saturate(dot(smoothNormal, sunDirection)) * targetInput.x * 4.0f; 
        float3 ambientLight = albedo * targetInput.y * 0.1f; 
        
        targetInput.xyz = saturate(directLight + ambientLight);
        targetInput.w = activations[featureQuartets + 4].w;
    }

    // Backpropagate error through MLP
    float4 errors[ACTIVATION_QUARTETS_PER_NETWORK];
    backpropLayer(targetInput, activations, errors, 4, 1, featureQuartets, featureQuartets + 4, outputOffset, OUTPUT_LAYER);   
    backpropLayer(targetInput, activations, errors, featureQuartets, 4, 0, featureQuartets, hiddenOffset, HIDDEN_LAYER);   
    
    // Distribute gradients to spatial features
    gateEncodingBackprop(gateData, errors);                                          

    // Accumulate MSE loss for GUI
    float pixelLoss = 0.0f;
    if (learningMode == 3)
    {
        float3 diffRGB = targetInput.xyz - activations[featureQuartets + 4].xyz;
        pixelLoss = dot(diffRGB, diffRGB) * 0.333f;
    }
    else
    {
        float2 diff2 = targetInput.xy - activations[featureQuartets + 4].xy; 
        pixelLoss = dot(diff2, diff2) * 0.5f; 
    }
    
    // Atomic float accumulation workaround
    LossBuffer.InterlockedAdd(0, (uint)(pixelLoss * 1000.0f));
}