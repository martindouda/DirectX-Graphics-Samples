struct GlobalTriangle
{
    uint i0, i1, i2;
    uint materialIdx;
};

struct GateFeature
{
    float4 data[2];
};

ByteAddressBuffer VertexBuffer                  : register(t0);
StructuredBuffer<GlobalTriangle> TriangleBuffer : register(t1);

StructuredBuffer<uint> VertexMaterialMap        : register(t2);
Texture2D<float4> MaterialTextures[]            : register(t0, space1);

RWStructuredBuffer<GateFeature> FeatureBuffer   : register(u0);
SamplerState LinearWrapSampler                  : register(s0);

cbuffer EncodeConstants : register(b0)
{
    uint TotalTriangles;
    uint VertexStride;
    uint UVOffset;
};

void assignColorToVertex(uint vertexIndex, uint textureIdx)
{
    uint uvAddress = (vertexIndex * VertexStride) + UVOffset;
    float2 uv = asfloat(VertexBuffer.Load2(uvAddress));
    float3 color = MaterialTextures[NonUniformResourceIndex(textureIdx)].SampleLevel(LinearWrapSampler, uv, 0).rgb;

    GateFeature feat = FeatureBuffer[vertexIndex];
    feat.data[0].xyz = color;
    feat.data[0].w = 1.0f;
    FeatureBuffer[vertexIndex] = feat;
}

[numthreads(64, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint triangleIndex = DTid.x;
    if (triangleIndex >= TotalTriangles)
        return;

    GlobalTriangle tri = TriangleBuffer[triangleIndex];
    uint textureIdx = tri.materialIdx * 6;

    assignColorToVertex(tri.i0, textureIdx);
    assignColorToVertex(tri.i1, textureIdx);
    assignColorToVertex(tri.i2, textureIdx);
}