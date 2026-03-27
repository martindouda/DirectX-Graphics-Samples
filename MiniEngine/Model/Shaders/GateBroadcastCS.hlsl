// GateBroadcastCS.hlsl

#include "GateTrainCommon.hlsli"

[numthreads(64, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint totalVertexID = DTid.x;

    if (totalVertexID >= totalTriangles * 3) // Nebo radìji pøedat totalVertices v cbufferu
        return;

    // 1. Zjistíme, do kterého unikátního bodu tento vertex fyzicky patøí
    uint uniqueID = VertexMappingBuffer[totalVertexID];

    // 2. Pøeèteme natrénovaný vektor z Unique bufferu
    GateFeature feature = UniqueFeatureBuffer[uniqueID];

    // 3. Zapíšeme do finálního bufferu pro inferenci
    DuplicatedFeatureBuffer[totalVertexID] = feature;
}