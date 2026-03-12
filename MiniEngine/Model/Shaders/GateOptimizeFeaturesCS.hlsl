// File: GateOptimizeFeaturesCS.hlsl

#include "GateTrainCommon.hlsli"

[numthreads(OPTIMIZATION_THREADGROUP_SIZE, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID) 
{
    uint index = DTid.x; 
    
    // We dispatch (TotalVertices * 2) threads.
    // Figure out which vertex we are modifying, and which of the two float4s it is.
    uint vertexIndex = index / 2;
    uint dataIndex = index % 2;

    // Unpack the fixed-point gradient accumulated during the backprop pass
    float4 gradient = unpackFloat4(GateFeatureGradientBuffer[index]);
    AdamData adam = GateFeatureAdamBuffer[index];
    
    // Read, add the Adam optimization step, and write directly back to the nested array
    GateFeatureBuffer[vertexIndex].data[dataIndex] += ApplyAdam(gradient, adam);
    
    // Save updated Adam state and zero out the gradient for the next training batch
    GateFeatureAdamBuffer[index] = adam;
    GateFeatureGradientBuffer[index] = int4(0, 0, 0, 0); 
}