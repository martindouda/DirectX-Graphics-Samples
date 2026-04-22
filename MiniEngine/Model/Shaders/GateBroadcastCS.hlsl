#include "GateTrainCommon.hlsli"

// Sparse push broadcast: copies updated unique features to all spatially identical geometric duplicates
[numthreads(BROADCAST_THREADGROUP_SIZE, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint uniqueID = DTid.x;
    
    // Prevent execution outside bounds and skip unmodified features to save memory bandwidth
    if (uniqueID >= uniqueVertexCount || FeatureDirtyBuffer[uniqueID] == 0) return;

    // Clear the dirty flag
    FeatureDirtyBuffer[uniqueID] = 0;

    uint duplicateCount = UniqueToDuplicateCountBuffer[uniqueID];
    uint duplicateOffset = UniqueToDuplicateOffsetBuffer[uniqueID];
    uint uniqueBase = uniqueID * featureQuartets;

    // Push optimized parameters to the unique buffer
    for (uint i = 0; i < duplicateCount; ++i)
    {
        uint duplicatedBase = DuplicateIndicesBuffer[duplicateOffset + i] * featureQuartets;
        
        for (uint q = 0; q < featureQuartets; ++q)
        {
            TargetFeatureBufferUAV[duplicatedBase + q] = UniqueFeatureBuffer[uniqueBase + q];
        }
    }
}