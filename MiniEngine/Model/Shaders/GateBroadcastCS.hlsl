// GateBroadcastCS.hlsl

#include "GateTrainCommon.hlsli"

[numthreads(64, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint duplicatedIndex = DTid.x;

    if (duplicatedIndex >= totalDuplicatedMeshColorPoints) 
        return;

    // 1. N:M Mapping: Find the unique point ID
    uint uniqueID = VertexMappingBuffer[duplicatedIndex]; // t3

    // 2. Read the newly optimized feature from Unique buffer
    GateFeature feature = UniqueFeatureBuffer[uniqueID];

    // 3. Broadcast it to the Duplicated buffer
    DuplicatedFeatureBuffer[duplicatedIndex] = feature;
}