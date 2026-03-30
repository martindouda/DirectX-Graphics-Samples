// GateBroadcastCS.hlsl

#include "GateTrainCommon.hlsli"

[numthreads(BROADCAST_THREADGROUP_SIZE, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint globalPointID = DTid.x;
    uint totalMeshColorPoints = totalTriangles * pointsPerTri;

    if (globalPointID >= totalMeshColorPoints)
        return;

    // 1. Zjistíme unikátní ID z našeho N:M mapování
    uint uniqueID = VertexMappingBuffer[globalPointID];

    // 2. Pøeèteme natrénovaný vektor a zapíšeme ho pro vykreslování
    GateFeature feature = UniqueFeatureBuffer[uniqueID];
    DuplicatedFeatureBuffer[globalPointID] = feature;
}