// File: GatePS.hlsl

#define GATE_INFERENCE
#include "GateTrainCommon.hlsli"


struct VSOutput 
{
    float4 Position  : SV_POSITION; 
    float2 UV        : TEXCOORD0; 
    float3 Normal    : NORMAL;
    float3 Tangent   : TANGENT;
    float3 Bitangent : BITANGENT;
    float3 worldPos  : WorldPos;
};

float4 main(VSOutput input, uint primitiveID : SV_PrimitiveID, float3 barycentrics : SV_Barycentrics) : SV_TARGET
{
    GlobalTriangle triData = GlobalTriangleBuffer[primitiveID + globalTriangleOffset];
    uint baseIndex = triData.pointOffset;
    uint localRes = triData.resolution;

    // Resolve structural grid indices and weights for the current pixel
    uint i0, j0, i1, j1, i2, j2;
    float weight0, weight1, weight2;
    getMeshColorIndicesAndWeights(barycentrics, localRes, i0, j0, weight0, i1, j1, weight1, i2, j2, weight2);

    uint flatIdx0 = (baseIndex + get1DIndex(i0, j0, localRes)) * featureQuartets;
    uint flatIdx1 = (baseIndex + get1DIndex(i1, j1, localRes)) * featureQuartets;
    uint flatIdx2 = (baseIndex + get1DIndex(i2, j2, localRes)) * featureQuartets;

    float4 activationsA[MAX_FEATURE_QUARTETS];
    float4 activationsB[MAX_FEATURE_QUARTETS];
    
    for (uint i = 0; i < MAX_FEATURE_QUARTETS; i++) 
    {
        activationsA[i] = 0.0f;
        activationsB[i] = 0.0f;
    }

    // Interpolate dynamic spatial features using structural barycentric coordinates
    for (uint q = 0; q < featureQuartets; ++q)
    { 
        float4 act = weight0 * FeatureBuffer[flatIdx0 + q] + 
                     weight1 * FeatureBuffer[flatIdx1 + q] + 
                     weight2 * FeatureBuffer[flatIdx2 + q];

        // Mask unused dimensionality bounds
        if (q * 4 + 0 >= featureFloats) act.x = 0.0f;
        if (q * 4 + 1 >= featureFloats) act.y = 0.0f;
        if (q * 4 + 2 >= featureFloats) act.z = 0.0f;
        if (q * 4 + 3 >= featureFloats) act.w = 0.0f;

        activationsA[q] = act;
    }

    // Execute Multi-Layer Perceptron (MLP) forward pass
    uint outputLayerOffset = (16 * featureQuartets) + 4;
    evalLayer(activationsA, activationsB, 0,                 4, featureQuartets, HIDDEN_LAYER);
    evalLayer(activationsB, activationsA, outputLayerOffset, 1, 4,               OUTPUT_LAYER);

    bool useTexturelessView      = (renderFlags & (1 << 0)) == 0;
    bool disableDirectionalLight = (renderFlags & (1 << 1)) == 0;

    // Early return to only show the grid
    bool showSubdivisionGrid = (renderFlags & (1 << 2)) != 0;
    if (showSubdivisionGrid)
    {
        float3 gridCoord = barycentrics * localRes;
        float3 edge = abs(gridCoord - round(gridCoord)) / fwidth(gridCoord);
        float lineIntensity = 1.0f - saturate(min(edge.x, min(edge.y, edge.z)) - 0.5f);
        return float4(lerp(float3(0.1f, 0.1f, 0.12f), float3(0.0f, 1.0f, 0.5f), lineIntensity), 1.0f);
    }

    // Early return to show the ground truth
    bool showGroundTruth = (renderFlags & (1 << 3)) != 0;
    if (showGroundTruth) 
    {
        // RNG per-pixel 
        uint pixelSeed = (uint)input.Position.x ^ pcgHash((uint)input.Position.y);
        uint rng = pcgHash(pixelSeed ^ pcgHash(frameIndex));

        float3 N = normalize(input.Normal);
        float3 worldPos = input.worldPos;

        // Trace Shadow Ray(s)
        float shadowAccumulation = 0.0f;
        float shadowMask = 1.0f;
        if ((lightingMode == 2 || lightingMode == 3))
        {
            for (uint s = 0; s < shadowSamples; ++s) // Using your UI-controlled cbuffer var
            {
                // Jitter the sun direction using our PRNG
                float3 jitteredSunDir = getConeSample(rand(rng), rand(rng), sunDirection, shadowSoftnessAngle);

                RayDesc shadowRay = { worldPos + N * 0.05f, 0.0f, jitteredSunDir, 10000.0f };
                RayQuery<RAY_FLAG_CULL_NON_OPAQUE | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> qShadow;
                qShadow.TraceRayInline(SceneBVH, 0, 0xFF, shadowRay);
                qShadow.Proceed();
            
                shadowAccumulation += (qShadow.CommittedStatus() == COMMITTED_TRIANGLE_HIT) ? 0.0f : 1.0f;
            }
            shadowMask = shadowAccumulation / max(1.0f, (float)shadowSamples);
        }

        // Trace AO Rays
        float aoAccumulation = 0.0f;
        float aoMask = 1.0f;
        if ((lightingMode == 1 || lightingMode == 3))
        {
            for (uint i = 0; i < aoSamples; ++i) // Use cbuffer var
            {
                float3 sampleDir = getCosineHemisphereSample(rand(rng), rand(rng), N);

                RayDesc aoRay = { worldPos + N * 0.05f, 0.0f, sampleDir, aoRadius }; // Use cbuffer var
                RayQuery<RAY_FLAG_CULL_NON_OPAQUE | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> qAO;
                qAO.TraceRayInline(SceneBVH, 0, 0xFF, aoRay);
                qAO.Proceed();
            
                aoAccumulation += (qAO.CommittedStatus() == COMMITTED_TRIANGLE_HIT) ? 0.0f : 1.0f;
            }
            aoMask = aoAccumulation / max(1.0f, (float)aoSamples);
        }

        // Sample Albedo
        float4 albedo = BindlessTextures[materialIdx * 6 + 0].Sample(LinearSampler, input.UV);
        if (useTexturelessView) albedo.rgb = float3(0.8f, 0.8f, 0.8f);

        // Calculate Multi-Sampled Lighting
        float fDiffuseLength = saturate(dot(N, normalize(sunDirection)));
        float3 directLight = 0.0f;
        if (!disableDirectionalLight)
            directLight = fDiffuseLength * albedo.rgb * (float3(1.0f, 1.0f, 1.0f) * sunIntensity) * shadowMask;
        float3 ambientColor = disableDirectionalLight ? float3(1.0f, 1.0f, 1.0f) : float3(0.1f, 0.1f, 0.1f); 
        float3 finalColor = directLight + (albedo.rgb * ambientColor * aoMask);

        return float4(finalColor, albedo.a);
    }

    // Output raw network predictions (RGB target) Note: Index is now 4!
    if (lightingMode == 4)
        return float4(saturate(activationsA[0].xyz), 1.0f);

    // Extract network predictions based on active visualization mode
    float shadowMask = (lightingMode == 2 || lightingMode == 3) ? saturate(activationsA[0].x) : 1.0f;
    float aoMask     = (lightingMode == 1 || lightingMode == 3) ? saturate(activationsA[0].y) : 1.0f;

    // Sample bindless material textures
    float4 albedo = BindlessTextures[materialIdx * 6 + 0].Sample(LinearSampler, input.UV);
    float specularMask = BindlessTextures[materialIdx * 6 + 1].Sample(LinearSampler, input.UV).g;
    
    if (useTexturelessView)
    {
        albedo.rgb = float3(0.8f, 0.8f, 0.8f);
        specularMask = 0.5f; 
    }
    
    // Evaluate tangent-space normal mapping
    float3 mapNormal = BindlessTextures[materialIdx * 6 + 3].Sample(LinearSampler, input.UV).rgb * 2.0f - 1.0f;
    float invNormalLen = rsqrt(dot(mapNormal, mapNormal));
    float gloss = lerp(1.0f, 128.0f, rcp(invNormalLen));

    float3x3 tbn = float3x3(normalize(input.Tangent), normalize(input.Bitangent), normalize(input.Normal));
    float3 N = normalize(mul(mapNormal * invNormalLen, tbn));

    // Evaluate physically-based analytical lighting model
    float3 L = normalize(sunDirection);
    float3 V = normalize(cameraPos - input.worldPos);
    float3 H = normalize(L + V);

    float fSpecularLength = specularMask * pow(saturate(dot(N, H)), gloss);
    float fDiffuseLength = saturate(dot(N, L));

    // Combine analytical lighting with implicit neural shadowing and occlusion
    float3 directLight = 0.0f;
    if (!disableDirectionalLight)
        directLight = (fDiffuseLength * albedo.rgb + fSpecularLength * float3(0.56f, 0.56f, 0.56f)) * (float3(1.0f, 1.0f, 1.0f) * sunIntensity) * shadowMask;
    
    float3 ambientColor = disableDirectionalLight ? float3(1.0f, 1.0f, 1.0f) : float3(0.1f, 0.1f, 0.1f); 
    float3 finalColor = directLight + (albedo.rgb * ambientColor * aoMask);

    return float4(finalColor, albedo.a);
}