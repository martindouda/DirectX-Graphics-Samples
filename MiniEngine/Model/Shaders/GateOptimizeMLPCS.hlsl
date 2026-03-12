// File: GateOptimizeMLPCS.hlsl

#include "GateTrainCommon.hlsli"


[numthreads(OPTIMIZATION_THREADGROUP_SIZE, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    uint index = DTid.x;
    // Assuming 41 total parameter quartets for 8->16->4
    if (index >= 41) return; 

    float4 gradient = unpackFloat4(MLPGradientBuffer[index]);
    AdamData adam = MLPAdamBuffer[index];
    
    MLPParameterBuffer[index] += ApplyAdam(gradient, adam);
    MLPAdamBuffer[index] = adam;
    MLPGradientBuffer[index] = 0; // Clear for next batch
}
