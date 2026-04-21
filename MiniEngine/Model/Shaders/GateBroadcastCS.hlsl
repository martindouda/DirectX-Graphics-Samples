// File: GateBroadcastCS.hlsl

#include "GateTrainCommon.hlsli"

// =========================================================================
//  KERNEL: Broadcast Unique Features to Duplicated Buffer
// =========================================================================

[numthreads(BROADCAST_THREADGROUP_SIZE, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint uniqueID = DTid.x;
    
    if (uniqueID >= uniqueVertexCount) 
        return;

    // If it did not change in this frame, skip the thread
    if (FeatureDirtyBuffer[uniqueID] == 0)
        return;

    // Reset the flag for the next frame (avoids the need for a full C++ ClearUAVUint)
    FeatureDirtyBuffer[uniqueID] = 0;

    uint duplicateCount = UniqueToDuplicateCountBuffer[uniqueID];
    uint duplicateOffset = UniqueToDuplicateOffsetBuffer[uniqueID];
    uint uniqueBase = uniqueID * featureQuartets;

    // "Push" architecture - upload new data to all its duplicates in the duplicates buffer
    for (uint i = 0; i < duplicateCount; ++i)
    {
        uint globalPointID = DuplicateIndicesBuffer[duplicateOffset + i];
        uint duplicatedBase = globalPointID * featureQuartets;

        for (uint q = 0; q < featureQuartets; ++q)
        {
            TargetFeatureBufferUAV[duplicatedBase + q] = UniqueFeatureBuffer[uniqueBase + q];
        }
    }
}