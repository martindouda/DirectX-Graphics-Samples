// File: GateOptimizeFeaturesCS.hlsl

#include "GateTrainCommon.hlsli"

// =========================================================================
//  KERNEL: Optimize Features (Adam Optimizer)
// =========================================================================

[numthreads(OPTIMIZATION_FEATURES_THREADGROUP_SIZE, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID) 
{
    uint index = DTid.x; 
    
    if (index >= uniqueVertexCount * featureQuartets)
        return;

    int4 packedGradient = FeatureGradientBuffer[index];
    
    if (packedGradient.x == 0 && packedGradient.y == 0 && packedGradient.z == 0 && packedGradient.w == 0)
        return;

    float4 gradient = unpackFloat4(packedGradient);
    gradient = clamp(gradient, -maxGradientClip, maxGradientClip);
    AdamData adam = FeatureAdamBuffer[index];
    
    float4 currentFeature = TargetFeatureBufferUAV[index];
    TargetFeatureBufferUAV[index] = currentFeature + ApplyAdam(gradient, currentFeature, adam, featureLearningRate, weightDecay);
    
    FeatureAdamBuffer[index] = adam;
    FeatureGradientBuffer[index] = int4(0, 0, 0, 0); 
}