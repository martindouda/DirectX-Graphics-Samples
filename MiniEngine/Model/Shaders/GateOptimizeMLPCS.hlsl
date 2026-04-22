#include "GateTrainCommon.hlsli"

// Applies Decoupled Weight Decay (AdamW) to the MLP parameters in parallel
[numthreads(OPTIMIZATION_MLP_THREADGROUP_SIZE, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID) 
{
    uint index = DTid.x;
    
    if (index >= mlpQuartets) return; 

    float4 gradient = clamp(unpackFloat4(MLPGradientBuffer[index]), -maxGradientClip, maxGradientClip);
    
    AdamData adam = MLPAdamBuffer[index];
    float4 currentWeight = MLPParameterBuffer[index];
    
    // Compute and apply the AdamW optimizer step using 1st/2nd moments
    MLPParameterBuffer[index] = currentWeight + ApplyAdam(gradient, currentWeight, adam, mlpLearningRate, weightDecay);
    
    // Store updated moments and zero out gradients for the next frame
    MLPAdamBuffer[index] = adam;
    MLPGradientBuffer[index] = int4(0, 0, 0, 0); 
}