// File: GateBroadcastCS.hlsl

#include "GateTrainCommon.hlsli"

// =========================================================================
//  KERNEL: Broadcast Unique Features to Duplicated Buffer
// =========================================================================

[numthreads(BROADCAST_THREADGROUP_SIZE, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint globalPointID = DTid.x;
    
    if (globalPointID >= totalMeshColorPoints) 
        return;

    uint uniqueID = VertexMappingBuffer[globalPointID];
    
    uint duplicatedBase = globalPointID * featureQuartets;
    uint uniqueBase = uniqueID * featureQuartets;

    for (uint q = 0; q < featureQuartets; ++q)
        TargetFeatureBufferUAV[duplicatedBase + q] = UniqueFeatureBuffer[uniqueBase + q];
}