struct GateFeature
{
    float4 data[2];
};

ByteAddressBuffer VertexBuffer                  : register(t0);
RWStructuredBuffer<GateFeature> FeatureBuffer   : register(u0);

cbuffer EncodeConstants : register(b0)
{
    uint TotalVertices;
    uint VertexStride;
    uint UVOffset;
    uint IsHalfFloat; // New flag!
};

[numthreads(64, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint vertexIndex = DTid.x;
    
    if (vertexIndex >= TotalVertices)
        return;

    uint uvAddress = (vertexIndex * VertexStride) + UVOffset;
    
    float2 uv;
    if (IsHalfFloat)
    {
        // Read 4 bytes total (two 16-bit floats)
        uint packedUV = VertexBuffer.Load(uvAddress);
        // Unpack lower 16 bits for X, upper 16 bits for Y
        uv = float2(f16tof32(packedUV), f16tof32(packedUV >> 16));
    }
    else
    {
        // Read 8 bytes total (two 32-bit floats)
        uv = asfloat(VertexBuffer.Load2(uvAddress));
    }

    GateFeature feat = FeatureBuffer[vertexIndex];
    feat.data[0].x = uv.x;
    feat.data[0].y = uv.y;
    FeatureBuffer[vertexIndex] = feat;
}