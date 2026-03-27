// File: GateOptimizeFeaturesCS.hlsl

#include "GateTrainCommon.hlsli"

[numthreads(OPTIMIZATION_THREADGROUP_SIZE, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID) 
{
    uint index = DTid.x; 
    
    int4 packedGradient = FeatureGradientBuffer[index];
    
    // If the gradient is 0 skip Adam entirely.
    if (packedGradient.x == 0 && packedGradient.y == 0 && packedGradient.z == 0 && packedGradient.w == 0)
        return;

    // TotalVertices * 2 threads dispatched. Each of the 2 handles a single quartet of the feature vector for a vertex
    uint vertexIndex = index / 2;
    uint dataIndex = index % 2;

    // Unpack the gradient accumulated during the backprop pass
    float4 gradient = unpackFloat4(packedGradient);
    AdamData adam = FeatureAdamBuffer[index];
    
    float4 currentFeature = DuplicatedFeatureBuffer[vertexIndex].data[dataIndex];
    DuplicatedFeatureBuffer[vertexIndex].data[dataIndex] += ApplyAdam(gradient, currentFeature, adam, featureLearningRate, weightDecay);
    
    // Save updated Adam state and zero out the gradient for the next training batch
    FeatureAdamBuffer[index] = adam;
    FeatureGradientBuffer[index] = int4(0, 0, 0, 0); 
}