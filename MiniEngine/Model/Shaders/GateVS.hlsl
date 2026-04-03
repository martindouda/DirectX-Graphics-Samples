// File: GateVS.hlsl

#define GATE_INFERENCE
#include "GateTrainCommon.hlsli"

// =========================================================================
//  Constant Buffers & Structures
// =========================================================================

cbuffer VSConstants : register(b0)
{
    float4x4 WVP;
};

struct VSInput 
{
    float3 position  : POSITION;
    float2 texcoord : TEXCOORD;
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

// =========================================================================
//  VERTEX SHADER: Forward Transformation
// =========================================================================

VSOutput main(VSInput input) 
{
    VSOutput output;
    
    output.position = mul(WVP, float4(input.position, 1.0f));
    output.worldPos = input.position;
    output.UV = input.texcoord;
    
    // Pass these through directly (No *2-1, they are FLOAT3)
    output.normal = input.normal;
    output.tangent = input.tangent;
    output.bitangent = input.bitangent;
    
    return output;
}