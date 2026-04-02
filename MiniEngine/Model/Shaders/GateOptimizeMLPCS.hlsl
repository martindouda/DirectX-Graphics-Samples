// File: GateOptimizeMLPCS.hlsl

#include "GateTrainCommon.hlsli"

// =========================================================================
//  KERNEL: Optimize MLP (Adam Optimizer)
// =========================================================================

[numthreads(OPTIMIZATION_MLP_THREADGROUP_SIZE, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID) 
{
    uint index = DTid.x;

    // The MLP has 212 parameters. Since we process them in quartets (float4), 
    // 212 / 4 = 53 quartets total.
    if (index >= 53) 
        return; 

    // Unpack the gradient accumulated during the backpropagation pass
    float4 gradient = unpackFloat4(MLPGradientBuffer[index]);
    AdamData adam = MLPAdamBuffer[index];
    
    // 1. Fetch the current MLP weights
    float4 currentWeight = MLPParameterBuffer[index];
    
    // 2. Apply the Adam optimization step
    MLPParameterBuffer[index] += ApplyAdam(gradient, currentWeight, adam, mlpLearningRate, weightDecay);

    // 3. Save the updated Adam state and zero out the gradient buffer for the next training step
    MLPAdamBuffer[index] = adam;
    MLPGradientBuffer[index] = int4(0, 0, 0, 0); 
}