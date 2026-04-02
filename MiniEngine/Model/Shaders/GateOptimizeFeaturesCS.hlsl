// File: GateOptimizeFeaturesCS.hlsl

#include "GateTrainCommon.hlsli"

// =========================================================================
//  KERNEL: Optimize Features (Adam Optimizer)
// =========================================================================

[numthreads(OPTIMIZATION_FEATURES_THREADGROUP_SIZE, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID) 
{
    uint index = DTid.x; 
    
    // Each unique vertex has 2 feature quartets (2 * float4)
    if (index >= uniqueVertexCount * 2)
        return;

    int4 packedGradient = FeatureGradientBuffer[index];
    
    // Early exit: If the gradient is exactly 0, skip the Adam update entirely
    if (packedGradient.x == 0 && packedGradient.y == 0 && packedGradient.z == 0 && packedGradient.w == 0)
        return;

    // Calculate vertex and data indices
    // index / 2 gives the unique vertex ID, index % 2 gives the feature quartet (0 or 1)
    uint vertexIndex = index / 2;
    uint dataIndex = index % 2;

    // Unpack the gradient accumulated during the backpropagation pass
    float4 gradient = unpackFloat4(packedGradient);
    AdamData adam = FeatureAdamBuffer[index];
    
    // Fetch the current feature and apply the Adam optimization step
    float4 currentFeature = DuplicatedFeatureBuffer[vertexIndex].data[dataIndex];
    DuplicatedFeatureBuffer[vertexIndex].data[dataIndex] += ApplyAdam(gradient, currentFeature, adam, featureLearningRate, weightDecay);
    
    // Save the updated Adam state and zero out the gradient buffer for the next training step
    FeatureAdamBuffer[index] = adam;
    FeatureGradientBuffer[index] = int4(0, 0, 0, 0); 
}