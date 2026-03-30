// File: GatePS.hlsl

#define GATE_INFERENCE
#include "GateTrainCommon.hlsli"

cbuffer MeshConstants : register(b1)
{
    uint globalTriangleOffset;
};

// Pøidáme náš buffer s trojúhelníky
StructuredBuffer<GlobalTriangle> TriangleBuffer : register(t2);

struct VSOutput 
{
    float4 Position : SV_POSITION;
};

// Pøidáváme SV_PrimitiveID a SV_Barycentrics
float4 main(VSOutput input, uint primitiveID : SV_PrimitiveID, float3 barycentrics : SV_Barycentrics) : SV_TARGET
{
    // 1. Najdeme globální ID trojúhelníku a vytáhneme si jeho definici
    uint globalTriID = primitiveID + globalTriangleOffset;
    GlobalTriangle tri = TriangleBuffer[globalTriID];

    // 2. Naèteme features pro všechny 3 vrcholy tohoto trojúhelníku
    GateFeature f0 = FeatureBuffer[tri.i0];
    GateFeature f1 = FeatureBuffer[tri.i1];
    GateFeature f2 = FeatureBuffer[tri.i2];

    // 3. Manuálnì nainterpolujeme features pomocí barycentrik
    float4 interpF0 = barycentrics.x * f0.data[0] + 
                      barycentrics.y * f1.data[0] + 
                      barycentrics.z * f2.data[0];
                      
    float4 interpF1 = barycentrics.x * f0.data[1] + 
                      barycentrics.y * f1.data[1] + 
                      barycentrics.z * f2.data[1];

    // --- Pùvodní logika architektury MLP zùstává ---
    float4 activationsA[MAX_NEURON_QUARTETS_PER_LAYER];
    float4 activationsB[MAX_NEURON_QUARTETS_PER_LAYER];

    activationsA[0] = interpF0;
    activationsA[1] = interpF1;
    activationsA[2] = 0.0f; 
    activationsA[3] = 0.0f;

    evalLayer(activationsA, activationsB, 0, 4, 2, HIDDEN_LAYER);
    evalLayer(activationsB, activationsA, 36, 1, 4, OUTPUT_LAYER);

    return float4(activationsA[0].xyz, 1.0f);
}