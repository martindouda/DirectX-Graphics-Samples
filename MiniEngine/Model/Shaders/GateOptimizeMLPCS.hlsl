// File: GateOptimizeMLPCS.hlsl

#include "GateTrainCommon.hlsli"

[numthreads(OPTIMIZATION_THREADGROUP_SIZE, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    uint index = DTid.x;

    if (index >= 53) return; // 212 floats / 4 = 53 quartets

    float4 gradient = unpackFloat4(MLPGradientBuffer[index]);
    AdamData adam = MLPAdamBuffer[index];
    
    // 1. Fetch the current MLP weight
    float4 currentWeight = MLPParameterBuffer[index];
    
    // 2. Pass it into ApplyAdam
    MLPParameterBuffer[index] += ApplyAdam(gradient, currentWeight, adam, mlpLearningRate, weightDecay);

    MLPAdamBuffer[index] = adam;
    MLPGradientBuffer[index] = 0; // Clear for next batch
}

