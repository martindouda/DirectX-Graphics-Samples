// File: GateVS.hlsl

#define GATE_INFERENCE
#include "GateTrainCommon.hlsli"

cbuffer VSConstants : register(b0)
{
    float4x4 WVP;
};

struct VSInput 
{
    float3 position  : POSITION;
    float2 texcoord  : TEXCOORD;
    float3 normal    : NORMAL;
    float3 tangent   : TANGENT;
    float3 bitangent : BITANGENT;
};

struct VSOutput 
{
    float4 position  : SV_POSITION;
    float2 UV        : TEXCOORD0;
    float3 normal    : NORMAL;
    float3 tangent   : TANGENT;
    float3 bitangent : BITANGENT;
    float3 worldPos  : WorldPos;
};

// Standard forward transformation vertex shader
VSOutput main(VSInput input) 
{
    VSOutput output;
    
    // Transform vertex position into clip space
    output.position = mul(WVP, float4(input.position, 1.0f));
    
    // Pass world-space position and un-transformed TBN vectors to the pixel shader
    output.worldPos = input.position;
    output.UV = input.texcoord;
    output.normal = input.normal;
    output.tangent = input.tangent;
    output.bitangent = input.bitangent;
    
    return output;
}