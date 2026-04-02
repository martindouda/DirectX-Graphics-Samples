// File: GatePS.hlsl

#define GATE_INFERENCE
#include "GateTrainCommon.hlsli"

cbuffer MeshConstants : register(b1) 
{ 
    uint globalTriangleOffset; 
    uint meshColorResolution;
    uint pointsPerTri;
    uint materialIdx;

    float3 sunDirection;
    float sunIntensity;
};

struct VSOutput 
{
    float4 Position : SV_POSITION; 
    float3 Normal   : NORMAL;
    float2 UV       : TEXCOORD0; // Opìt, ujisti se, že VS posílá UV
};

float4 main(VSOutput input, uint primitiveID : SV_PrimitiveID, float3 barycentrics : SV_Barycentrics) : SV_TARGET
{
    // ==========================================================
    // 1. INFERENCE NEURONOVÉ SÍTÌ
    // ==========================================================
    uint globalTriID = primitiveID + globalTriangleOffset;
    uint baseIndex = globalTriID * pointsPerTri;

    uint i0, j0, i1, j1, i2, j2;
    float weight0, weight1, weight2;
    
    getMeshColorIndicesAndWeights(barycentrics, meshColorResolution, i0, j0, weight0, i1, j1, weight1, i2, j2, weight2);

    uint idx0 = get1DIndex(i0, j0, meshColorResolution);
    uint idx1 = get1DIndex(i1, j1, meshColorResolution);
    uint idx2 = get1DIndex(i2, j2, meshColorResolution);

    GateFeature f0 = FeatureBuffer[baseIndex + idx0];
    GateFeature f1 = FeatureBuffer[baseIndex + idx1];
    GateFeature f2 = FeatureBuffer[baseIndex + idx2];

    float4 interpF0 = weight0 * f0.data[0] + weight1 * f1.data[0] + weight2 * f2.data[0];
    float4 interpF1 = weight0 * f0.data[1] + weight1 * f1.data[1] + weight2 * f2.data[1];

    float4 activationsA[MAX_NEURON_QUARTETS_PER_LAYER];
    float4 activationsB[MAX_NEURON_QUARTETS_PER_LAYER];
    activationsA[0] = interpF0; activationsA[1] = interpF1; activationsA[2] = 0.0f; activationsA[3] = 0.0f;

    evalLayer(activationsA, activationsB, 0, 4, 2, HIDDEN_LAYER);
    evalLayer(activationsB, activationsA, 36, 1, 4, OUTPUT_LAYER);

    float shadowMask = lerp(0.1f, 1.0f, activationsA[0].x); 
    
    // ==========================================================
    // 2. TEXTURA A JEDNODUCHÉ NASVÍCENÍ
    // ==========================================================
    float4 albedo = BindlessTextures[materialIdx * 6].Sample(LinearSampler, input.UV);
    
    // Normalizace vektorù pro jistotu
    float3 N = normalize(input.Normal);
    float3 L = normalize(sunDirection);
    
    // Skalární souèin: èím víc je normála pøiklonìna ke slunci, tím je hodnota blíž 1.0
    // saturate() oøízne záporné hodnoty na 0 (když slunce svítí zezadu)
    float NdotL = saturate(dot(N, L));
    
    // Pøímé svìtlo (Slunce * Úhel dopadu * Stín ze sítì)
    float3 directLight = albedo.rgb * NdotL * sunIntensity * shadowMask;
    
    // Ambientní svìtlo (aby odvrácené strany a stíny nebyly absolutnì èerné)
    float3 ambientLight = albedo.rgb * 0.1f;
    
    return float4(directLight + ambientLight, albedo.a);
}