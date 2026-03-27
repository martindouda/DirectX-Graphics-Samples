// File: GateVS.hlsl

#define GATE_INFERENCE
#include "GateTrainCommon.hlsli"

cbuffer VSConstants : register(b0)
{
    float4x4 WVP;
};

cbuffer MeshConstants : register(b1)
{
    uint globalMeshOffset;
};

struct VSInput 
{
    float3 position : POSITION;
};

struct VSOutput 
{
    float4 position : SV_POSITION;
    float4 f0 : TEXCOORD0;
    float4 f1 : TEXCOORD1;
};

VSOutput main(VSInput input, uint vertexID : SV_VertexID) 
{
    VSOutput output;
    output.position = mul(WVP, float4(input.position, 1.0f));

    GateFeature feat = FeatureBuffer[vertexID + globalMeshOffset];

    output.f0 = feat.data[0];
    output.f1 = feat.data[1];

    return output;
}