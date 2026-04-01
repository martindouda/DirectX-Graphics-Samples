// VisBufferCS.hlsl

Texture2D<uint> VisibilityBuffer : register(t0);
RWTexture2D<float4> VisOutput    : register(u0);

// Your existing hash function
uint pcgHash(uint v)
{
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float3 hashToColor(uint hash)
{
    // Extract 3 bytes to make an RGB color
    return float3(
        (hash & 255) / 255.0f,
        ((hash >> 8) & 255) / 255.0f,
        ((hash >> 16) & 255) / 255.0f
    );
}

[numthreads(8, 8, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint width, height;
    VisibilityBuffer.GetDimensions(width, height);
    if (DTid.x >= width || DTid.y >= height) return;

    uint triID = VisibilityBuffer[DTid.xy];

    if (triID == 0) 
    {
        VisOutput[DTid.xy] = float4(0.1f, 0.1f, 0.1f, 1.0f); // Dark grey for sky
    }
    else
    {
        uint hash = pcgHash(triID);
        VisOutput[DTid.xy] = float4(hashToColor(hash), 1.0f);
    }
}