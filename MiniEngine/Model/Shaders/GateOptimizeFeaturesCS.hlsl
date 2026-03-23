// File: GateOptimizeFeaturesCS.hlsl

#include "GateTrainCommon.hlsli"

[numthreads(OPTIMIZATION_THREADGROUP_SIZE, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID) 
{
    uint index = DTid.x; 
    
    int4 packedGradient = GateFeatureGradientBuffer[index];
    
    // BOKŠANSKÝ'S TRICK: If the gradient is perfectly 0, no backprop thread 
    // touched this feature this frame. Skip Adam entirely!
    if (packedGradient.x == 0 && packedGradient.y == 0 && packedGradient.z == 0 && packedGradient.w == 0)
        return;

    // We dispatch (TotalVertices * 2) threads.
    // Figure out which vertex we are modifying, and which of the two float4s it is.
    uint vertexIndex = index / 2;
    uint dataIndex = index % 2;

    // Unpack the fixed-point gradient accumulated during the backprop pass
    float4 gradient = unpackFloat4(packedGradient);
    AdamData adam = GateFeatureAdamBuffer[index];
    
    // Read, add the Adam optimization step, and write directly back to the nested array
    GateFeatureBuffer[vertexIndex].data[dataIndex] += ApplyAdam(gradient, adam, featureLearningRate);
    
    // Save updated Adam state and zero out the gradient for the next training batch
    GateFeatureAdamBuffer[index] = adam;
    GateFeatureGradientBuffer[index] = int4(0, 0, 0, 0); 
}