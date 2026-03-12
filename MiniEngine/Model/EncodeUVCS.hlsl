struct GateFeature
{
    float4 data[2];
};

ByteAddressBuffer VertexBuffer                  : register(t0);
StructuredBuffer<uint> VertexMaterialMap        : register(t1);
Texture2D<float4> MaterialTextures[]            : register(t0, space1);

RWStructuredBuffer<GateFeature> FeatureBuffer   : register(u0);
SamplerState LinearWrapSampler                  : register(s0);

cbuffer EncodeConstants : register(b0)
{
    uint TotalVertices;
    uint VertexStride;
    uint UVOffset;
};

[numthreads(64, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint vertexIndex = DTid.x;
    if (vertexIndex >= TotalVertices)
        return;

    uint uvAddress = (vertexIndex * VertexStride) + UVOffset;
    float2 uv = asfloat(VertexBuffer.Load2(uvAddress));

    uint materialIdx = VertexMaterialMap[vertexIndex];
    uint textureIdx = materialIdx * 6;
    float3 color = MaterialTextures[NonUniformResourceIndex(textureIdx)].SampleLevel(LinearWrapSampler, uv, 0).rgb;

    GateFeature feat = FeatureBuffer[vertexIndex];
    feat.data[0].xyz = color;
    feat.data[0].w = 1.0f;
    FeatureBuffer[vertexIndex] = feat;
}