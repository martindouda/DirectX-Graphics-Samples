// File: GateOptimizeFeaturesCS.hlsl

#include "GateTrainCommon.hlsli"

[numthreads(OPTIMIZATION_THREADGROUP_SIZE, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID) 
{
    uint index = DTid.x; 
    
    if (index >= totalUniqueMeshColorPoints * 2) 
        return;

    int4 packedGradient = FeatureGradientBuffer[index];
    
    if (packedGradient.x == 0 && packedGradient.y == 0 && packedGradient.z == 0 && packedGradient.w == 0)
        return;

    uint uniqueIndex = index / 2;
    uint dataIndex = index % 2;

    float4 gradient = unpackFloat4(packedGradient);
    AdamData adam = FeatureAdamBuffer[index];
    
    // NOTE: We use "DuplicatedFeatureBuffer" here because it is the alias for register(u0).
    // During this pass, C++ binds m_UniqueFeatureBuffer to u0, so this perfectly optimizes the unique points!
    float4 currentFeature = DuplicatedFeatureBuffer[uniqueIndex].data[dataIndex];
    DuplicatedFeatureBuffer[uniqueIndex].data[dataIndex] += ApplyAdam(gradient, currentFeature, adam, featureLearningRate, weightDecay);
    
    FeatureAdamBuffer[index] = adam;
    FeatureGradientBuffer[index] = int4(0, 0, 0, 0); 
}