#define GATE_INFERENCE
#include "GateTrainCommon.hlsli"

cbuffer MeshConstants : register(b1) { uint globalTriangleOffset; };

struct VSOutput { float4 Position : SV_POSITION; };

float4 main(VSOutput input, uint primitiveID : SV_PrimitiveID, float3 barycentrics : SV_Barycentrics) : SV_TARGET
{
    uint globalTriID = primitiveID + globalTriangleOffset;
    uint baseIndex = globalTriID * 6; // Každý trojúhelník má 6 vektorù

    float u = barycentrics.x * 2.0f; float v = barycentrics.y * 2.0f; float w = barycentrics.z * 2.0f;
    uint i0, i1, i2; float weight0, weight1, weight2;

    if (u >= 1.0f) { i0 = 0; i1 = 3; i2 = 5; weight0 = u - 1.0f; weight1 = v; weight2 = w; }
    else if (v >= 1.0f) { i0 = 3; i1 = 1; i2 = 4; weight0 = u; weight1 = v - 1.0f; weight2 = w; }
    else if (w >= 1.0f) { i0 = 5; i1 = 4; i2 = 2; weight0 = u; weight1 = v; weight2 = w - 1.0f; }
    else { i0 = 3; i1 = 4; i2 = 5; weight0 = 1.0f - w; weight1 = 1.0f - u; weight2 = 1.0f - v; }

    // Naèteme 3 správné vektory a nainterpolujeme je
    GateFeature f0 = FeatureBuffer[baseIndex + i0];
    GateFeature f1 = FeatureBuffer[baseIndex + i1];
    GateFeature f2 = FeatureBuffer[baseIndex + i2];

    float4 interpF0 = weight0 * f0.data[0] + weight1 * f1.data[0] + weight2 * f2.data[0];
    float4 interpF1 = weight0 * f0.data[1] + weight1 * f1.data[1] + weight2 * f2.data[1];

    float4 activationsA[MAX_NEURON_QUARTETS_PER_LAYER];
    float4 activationsB[MAX_NEURON_QUARTETS_PER_LAYER];
    activationsA[0] = interpF0; activationsA[1] = interpF1; activationsA[2] = 0.0f; activationsA[3] = 0.0f;

    evalLayer(activationsA, activationsB, 0, 4, 2, HIDDEN_LAYER);
    evalLayer(activationsB, activationsA, 36, 1, 4, OUTPUT_LAYER);

    return float4(activationsA[0].xyz, 1.0f);
}