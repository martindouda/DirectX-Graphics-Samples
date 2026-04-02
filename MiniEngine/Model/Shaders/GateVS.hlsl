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
    float3 position : POSITION;
    float3 normal   : NORMAL;
    float2 UV       : TEXCOORD0;
};

struct VSOutput 
{
    float4 position : SV_POSITION;
    float3 normal   : NORMAL;
    float2 UV       : TEXCOORD0;
};

// =========================================================================
//  VERTEX SHADER: Forward Transformation
// =========================================================================

VSOutput main(VSInput input) 
{
    VSOutput output;
    
    // Transform the vertex position into clip space
    output.position = mul(WVP, float4(input.position, 1.0f));
    
    // Pass-through the normal and UV coordinates to the pixel shader
    output.normal = input.normal;
    output.UV = input.UV;
    
    return output;
}