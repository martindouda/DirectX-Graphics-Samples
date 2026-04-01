// DepthViewerPS.hlsl

//
// Copyright (c) Microsoft. All rights reserved.
// This code is licensed under the MIT License (MIT).
// THIS CODE IS PROVIDED *AS IS* WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING ANY
// IMPLIED WARRANTIES OF FITNESS FOR A PARTICULAR
// PURPOSE, MERCHANTABILITY, OR NON-INFRINGEMENT.
//
// Developed by Minigraph
//
// Author(s):	James Stanard
//

#include "Common.hlsli"

// Match the struct we created for the C++ RenderObjects loop
cbuffer MeshConstants : register(b1)
{
    uint materialIdx;
    uint globalTriangleOffset;
};

struct VSOutput
{
    float4 pos : SV_Position;
    float2 uv : TexCoord0;
};

Texture2D<float4> texDiffuse : register(t0);

[RootSignature(Renderer_RootSig)]
uint main(VSOutput vsOutput, uint primitiveID : SV_PrimitiveID) : SV_Target0
{
    // If we are compiling the Cutout version of this shader, do the alpha test
#ifdef CUTOUT
    if (texDiffuse.Sample(defaultSampler, vsOutput.uv).a < 0.5)
        discard;
#endif

    // Calculate and return the global triangle ID
    return globalTriangleOffset + primitiveID + 1;
}