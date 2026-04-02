// File: GateBroadcastCS.hlsl

#include "GateTrainCommon.hlsli"

// =========================================================================
//  KERNEL: Broadcast Unique Features to Duplicated Buffer
// =========================================================================

[numthreads(BROADCAST_THREADGROUP_SIZE, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint globalPointID = DTid.x;
    uint totalMeshColorPoints = totalTriangles * pointsPerTri;

    if (globalPointID >= totalMeshColorPoints)
        return;

    // 1. Retrieve the unique ID from our N:M mapping buffer
    uint uniqueID = VertexMappingBuffer[globalPointID];

    // 2. Read the trained feature vector and write it to the duplicated buffer for rendering
    GateFeature feature = UniqueFeatureBuffer[uniqueID];
    DuplicatedFeatureBuffer[globalPointID] = feature;
}