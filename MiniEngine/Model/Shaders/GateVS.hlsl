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
    float2 UV       : TEXCOORD0;
};

struct VSOutput 
{
    float4 position : SV_POSITION;
    float2 UV	    : TEXCOORD0;
};

VSOutput main(VSInput input) 
{
    VSOutput output;
    output.position = mul(WVP, float4(input.position, 1.0f));
    output.UV = input.UV;
    return output;
}