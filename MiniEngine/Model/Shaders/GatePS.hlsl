// File: GatePS.hlsl

#define GATE_INFERENCE
#include "GateTrainCommon.hlsli"

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
    float2 pad;

    float3 cameraPos;
    float pad2;
};

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
    uint globalTriID = primitiveID + globalTriangleOffset;
    
    GlobalTriangle triData = GlobalTriangleBuffer[globalTriID];
    uint baseIndex = triData.pointOffset;
    uint localRes = triData.resolution;

    uint i0, j0, i1, j1, i2, j2;
    float weight0, weight1, weight2;
    
    getMeshColorIndicesAndWeights(barycentrics, localRes, i0, j0, weight0, i1, j1, weight1, i2, j2, weight2);

    uint flatIdx0 = (baseIndex + get1DIndex(i0, j0, localRes)) * featureQuartets;
    uint flatIdx1 = (baseIndex + get1DIndex(i1, j1, localRes)) * featureQuartets;
    uint flatIdx2 = (baseIndex + get1DIndex(i2, j2, localRes)) * featureQuartets;

    float4 activationsA[MAX_FEATURE_QUARTETS];
    float4 activationsB[MAX_FEATURE_QUARTETS];
    
    for (uint i = 0; i < MAX_FEATURE_QUARTETS; i++) {
        activationsA[i] = 0.0f;
        activationsB[i] = 0.0f;
    }

    // Dynamic Network Input Interpolation
    for (uint q = 0; q < featureQuartets; ++q)
    { 
        float4 d0 = FeatureBuffer[flatIdx0 + q];
        float4 d1 = FeatureBuffer[flatIdx1 + q];
        float4 d2 = FeatureBuffer[flatIdx2 + q];
        
        float4 act = weight0 * d0 + weight1 * d1 + weight2 * d2;

        if (q * 4 + 0 >= featureFloats) act.x = 0.0f;
        if (q * 4 + 1 >= featureFloats) act.y = 0.0f;
        if (q * 4 + 2 >= featureFloats) act.z = 0.0f;
        if (q * 4 + 3 >= featureFloats) act.w = 0.0f;

        activationsA[q] = act;
    }

    // Dynamic Layer Evaluation
    uint outputLayerOffset = (16 * featureQuartets) + 4;
    evalLayer(activationsA, activationsB, 0,                 4, featureQuartets, HIDDEN_LAYER);
    evalLayer(activationsB, activationsA, outputLayerOffset, 1, 4,               OUTPUT_LAYER);

    float networkShadow = saturate(activationsA[0].x); 
    float networkAO     = saturate(activationsA[0].y); 
    
    float shadowMask = 1.0f;
    float aoMask = 1.0f;
    
    if (lightingMode == 1 || lightingMode == 3) aoMask = networkAO;
    if (lightingMode == 2 || lightingMode == 3) shadowMask = networkShadow;
    
    bool useTexturelessView      = (renderFlags & (1 << 0)) != 0;
    bool disableDirectionalLight = (renderFlags & (1 << 1)) != 0;

    // --- TEXTURE SAMPLING ---
    float4 albedo = BindlessTextures[materialIdx * 6 + 0].Sample(LinearSampler, input.UV);
    float specularMask = BindlessTextures[materialIdx * 6 + 1].Sample(LinearSampler, input.UV).g;
    
    if (useTexturelessView)
    {
        albedo.rgb = float3(0.8f, 0.8f, 0.8f);
        specularMask = 0.5f; 
    }
    
    // --- MINIENGINE EXACT NORMAL & TOKSVIG AA ---
    float gloss = 128.0f;
    float3 mapNormal = BindlessTextures[materialIdx * 6 + 3].Sample(LinearSampler, input.UV).rgb;
    mapNormal = mapNormal * 2.0f - 1.0f;

    // Detect mipmap degradation and soften the gloss automatically
    float normalLenSq = dot(mapNormal, mapNormal);
    float invNormalLen = rsqrt(normalLenSq);
    mapNormal *= invNormalLen;
    gloss = lerp(1.0f, gloss, rcp(invNormalLen));

    float3x3 tbn = float3x3(normalize(input.Tangent), normalize(input.Bitangent), normalize(input.Normal));
    float3 N = normalize(mul(mapNormal, tbn));

    // --- MINIENGINE EXACT LIGHTING MATH ---
    float3 specularAlbedo = float3(0.56f, 0.56f, 0.56f);
    float3 L = normalize(sunDirection);
    float3 V = normalize(cameraPos - input.worldPos);
    float3 H = normalize(L + V);

    float NdotL = saturate(dot(N, L));
    float NdotH = saturate(dot(N, H));

    float fSpecularLength = specularMask * pow(NdotH, gloss);
    float fDiffuseLength = NdotL;

    float3 sunColor = float3(1.0f, 1.0f, 1.0f) * sunIntensity;
    
    float3 directLight = 0.0f;
    if (!disableDirectionalLight)
    {
        // Notice: fSpecularLength is NOT multiplied by NdotL here, matching MiniEngine exactly
        directLight = (fDiffuseLength * albedo.rgb + fSpecularLength * specularAlbedo) * sunColor * shadowMask;
    }
    
    // MiniEngine Sponza defaults to exactly 0.1 ambient intensity
    float3 ambientColor = float3(0.1f, 0.1f, 0.1f); 
    if (disableDirectionalLight) ambientColor = float3(1.0f, 1.0f, 1.0f); // Boost if sun is off
    
    float3 ambientLight = albedo.rgb * ambientColor * aoMask; 
    
    float3 finalColor = directLight + ambientLight;

    return float4(finalColor, albedo.a);
}