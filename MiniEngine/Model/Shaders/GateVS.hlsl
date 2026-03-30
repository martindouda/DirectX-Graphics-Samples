// File: GateVS.hlsl

#define GATE_INFERENCE
#include "GateTrainCommon.hlsli"

cbuffer VSConstants : register(b0)
{
    float4x4 WVP;
};

struct VSInput 
{
    float3 position : POSITION;
};

struct VSOutput 
{
    float4 position : SV_POSITION;
};

VSOutput main(VSInput input) 
{
    VSOutput output;
    output.position = mul(WVP, float4(input.position, 1.0f));
    return output;
}