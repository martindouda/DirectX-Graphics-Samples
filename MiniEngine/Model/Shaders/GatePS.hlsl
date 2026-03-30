// File: GatePS.hlsl

#define GATE_INFERENCE
#include "GateTrainCommon.hlsli"

cbuffer MeshConstants : register(b1) 
{ 
    uint globalTriangleOffset; 
    uint meshColorResolution;
    uint pointsPerTri;
};

struct VSOutput 
{
    float4 Position : SV_POSITION; 
};

float4 main(VSOutput input, uint primitiveID : SV_PrimitiveID, float3 barycentrics : SV_Barycentrics) : SV_TARGET
{
    uint globalTriID = primitiveID + globalTriangleOffset;
    uint baseIndex = globalTriID * pointsPerTri;

    uint i0, j0, i1, j1, i2, j2;
    float weight0, weight1, weight2;
    
    // Získáme lokální møížkové souøadnice
    getMeshColorIndicesAndWeights(barycentrics, meshColorResolution, i0, j0, weight0, i1, j1, weight1, i2, j2, weight2);

    // Pøevedeme na 1D indexy
    uint idx0 = get1DIndex(i0, j0, meshColorResolution);
    uint idx1 = get1DIndex(i1, j1, meshColorResolution);
    uint idx2 = get1DIndex(i2, j2, meshColorResolution);

    // Naèteme 3 správné vektory a nainterpolujeme je
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

    return float4(activationsA[0].xyz, 1.0f);
}