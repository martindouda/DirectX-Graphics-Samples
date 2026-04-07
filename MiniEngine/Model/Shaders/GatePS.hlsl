// File: GatePS.hlsl

#define GATE_INFERENCE
#include "GateTrainCommon.hlsli"

cbuffer MeshConstants : register(b1) 
{ 
    uint globalTriangleOffset; 
    uint lightingMode;
    uint renderFlags;
    uint materialIdx;
    uint featureQuartets; // Dynamic feature scaling constant

    float3 sunDirection;
    float sunIntensity;
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

float3 ComputeNormal(VSOutput input)
{
    float3 mapNormal = BindlessTextures[materialIdx * 6 + 3].Sample(LinearSampler, input.UV).rgb;
    mapNormal = mapNormal * 2.0 - 1.0;
    float3x3 tbn = float3x3(normalize(input.Tangent), normalize(input.Bitangent), normalize(input.Normal));
    return normalize(mul(mapNormal, tbn));
}

float4 main(VSOutput input, uint primitiveID : SV_PrimitiveID, float3 barycentrics : SV_Barycentrics) : SV_TARGET
{
    uint globalTriID = primitiveID + globalTriangleOffset;
    
    GlobalTriangle triData = GlobalTriangleBuffer[globalTriID];
    uint baseIndex = triData.pointOffset;
    uint localRes = triData.resolution;

    uint i0, j0, i1, j1, i2, j2;
    float weight0, weight1, weight2;
    
    getMeshColorIndicesAndWeights(barycentrics, localRes, i0, j0, weight0, i1, j1, weight1, i2, j2, weight2);

    // Calculate flat array offsets dynamically.
    // Instead of using structs, the buffer is flattened. To find the exact float4
    // memory block for a vertex, multiply the structural index by featureQuartets.
    uint flatIdx0 = (baseIndex + get1DIndex(i0, j0, localRes)) * featureQuartets;
    uint flatIdx1 = (baseIndex + get1DIndex(i1, j1, localRes)) * featureQuartets;
    uint flatIdx2 = (baseIndex + get1DIndex(i2, j2, localRes)) * featureQuartets;

    // Use MAX_FEATURE_QUARTETS to establish the upper memory bound. 
    // This prevents compilation errors and out-of-bound GPU accesses.
    float4 activationsA[MAX_FEATURE_QUARTETS];
    float4 activationsB[MAX_FEATURE_QUARTETS];
    
    for(uint i = 0; i < MAX_FEATURE_QUARTETS; i++) {
        activationsA[i] = 0.0f;
        activationsB[i] = 0.0f;
    }

    // Dynamic Network Input Interpolation
    for(uint q = 0; q < featureQuartets; ++q)
    {
        float4 d0 = FeatureBuffer[flatIdx0 + q];
        float4 d1 = FeatureBuffer[flatIdx1 + q];
        float4 d2 = FeatureBuffer[flatIdx2 + q];
        activationsA[q] = weight0 * d0 + weight1 * d1 + weight2 * d2;
    }

    // Dynamic Layer Evaluation
    // Calculate exact offset into the MLP buffer based on the dynamic feature size
    uint outputLayerOffset = (16 * featureQuartets) + 4;
    evalLayer(activationsA, activationsB, 0,                 4, featureQuartets, HIDDEN_LAYER);
    evalLayer(activationsB, activationsA, outputLayerOffset, 1, 4,               OUTPUT_LAYER);

    float networkShadow = saturate(activationsA[0].x); 
    float networkAO     = saturate(activationsA[0].y); 
    
    float shadowMask = 1.0f;
    float aoMask = 1.0f;
    
    // Lighting Mode overrides
    if (lightingMode == 1 || lightingMode == 3)
        aoMask = networkAO;
        
    if (lightingMode == 2 || lightingMode == 3)
        shadowMask = networkShadow;
    
    bool useTexturelessView      = (renderFlags & (1 << 0)) != 0;
    bool disableDirectionalLight = (renderFlags & (1 << 1)) != 0;

    float4 albedo = BindlessTextures[materialIdx * 6 + 0].Sample(LinearSampler, input.UV);
    
    if (useTexturelessView)
    {
        albedo.rgb = float3(0.8f, 0.8f, 0.8f);
    }
    
    float3 N = ComputeNormal(input);
    float3 L = normalize(sunDirection);
    float NdotL = saturate(dot(N, L));
    
    float3 directLight = 0.0f;
    if (!disableDirectionalLight)
        directLight = albedo.rgb * NdotL * sunIntensity * shadowMask; 
    
    float ambientBase = disableDirectionalLight ? 1.0f : 0.4f;
    float3 ambientLight = albedo.rgb * ambientBase * aoMask; 
    
    return float4(directLight + ambientLight, albedo.a);
}