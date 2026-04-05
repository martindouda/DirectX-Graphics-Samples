// File: GateBroadcastCS.hlsl

#include "GateTrainCommon.hlsli"

// =========================================================================
//  KERNEL: Broadcast Unique Features to Duplicated Buffer
// =========================================================================

[numthreads(BROADCAST_THREADGROUP_SIZE, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint globalPointID = DTid.x;
    
    // RootConstants CB needs to pass totalMeshColorPoints
    if (globalPointID >= totalMeshColorPoints) 
        return;

    uint uniqueID = VertexMappingBuffer[globalPointID];
    GateFeature feature = UniqueFeatureBuffer[uniqueID];
    DuplicatedFeatureBuffer[globalPointID] = feature;
}