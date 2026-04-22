#include "GateTrainCommon.hlsli"

// Applies Decoupled Weight Decay (AdamW) to the unique spatial feature grid in parallel
[numthreads(OPTIMIZATION_FEATURES_THREADGROUP_SIZE, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID) 
{
    uint index = DTid.x; 
    
    if (index >= uniqueVertexCount * featureQuartets) return;

    // Early exit to skip unmodified features (saves memory bandwidth and Adam moment drift)
    int4 packedGradient = FeatureGradientBuffer[index];
    if (all(packedGradient == 0)) return;

    // Flag this spatial feature as dirty to trigger a sparse broadcast to its geometric duplicates
    uint uniqueID = index / featureQuartets;
    FeatureDirtyBuffer[uniqueID] = 1;

    float4 gradient = clamp(unpackFloat4(packedGradient), -maxGradientClip, maxGradientClip);
    
    AdamData adam = FeatureAdamBuffer[index];
    float4 currentFeature = TargetFeatureBufferUAV[index];
    
    // Compute and apply the AdamW optimizer step using 1st/2nd moments
    TargetFeatureBufferUAV[index] = currentFeature + ApplyAdam(gradient, currentFeature, adam, featureLearningRate, weightDecay);
    
    // Store updated moments and zero out gradients for the next frame's backpropagation
    FeatureAdamBuffer[index] = adam;
    FeatureGradientBuffer[index] = int4(0, 0, 0, 0); 
}