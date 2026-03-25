// GateBroadcastCS.hlsl
#include "GateTrainCommon.hlsli"

// SRV: Pole intù, kde index je TotalVertexID a hodnota je UniqueVertexID
StructuredBuffer<uint> VertexMappingBuffer : register(t3);

// SRV: Natrénované unikátní hodnoty (velikost: m_UniqueSpatialVertexCount)
StructuredBuffer<GateFeature> UniqueFeatureBuffer_SRV : register(t4);

// UAV: Finální buffer pro renderování (velikost: m_TotalVertices)
RWStructuredBuffer<GateFeature> DuplicatedFeatureBuffer_UAV : register(u0);

[numthreads(64, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint totalVertexID = DTid.x;

    if (totalVertexID >= totalTriangles * 3) // Nebo radìji pøedat totalVertices v cbufferu
        return;

    // 1. Zjistíme, do kterého unikátního bodu tento vertex fyzicky patøí
    uint uniqueID = VertexMappingBuffer[totalVertexID];

    // 2. Pøeèteme natrénovaný vektor z Unique bufferu
    GateFeature feature = UniqueFeatureBuffer_SRV[uniqueID];

    // 3. Zapíšeme do finálního bufferu pro inferenci
    DuplicatedFeatureBuffer_UAV[totalVertexID] = feature;
}