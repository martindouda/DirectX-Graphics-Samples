// File: GateOptimizeMLPCS.hlsl

#include "GateTrainCommon.hlsli"

// =========================================================================
//  KERNEL: Optimize MLP (Adam Optimizer)
// =========================================================================

[numthreads(OPTIMIZATION_MLP_THREADGROUP_SIZE, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID) 
{
    uint index = DTid.x;

    if (index >= mlpQuartets) 
        return; 

    float4 gradient = unpackFloat4(MLPGradientBuffer[index]);
    AdamData adam = MLPAdamBuffer[index];
    
    float4 currentWeight = MLPParameterBuffer[index];
    MLPParameterBuffer[index] = currentWeight + ApplyAdam(gradient, currentWeight, adam, mlpLearningRate, weightDecay);

    MLPAdamBuffer[index] = adam;
    MLPGradientBuffer[index] = int4(0, 0, 0, 0); 
}