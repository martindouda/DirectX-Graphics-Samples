// Copyright (c) Microsoft. All rights reserved.
// This code is licensed under the MIT License (MIT).

#include "Common.hlsli"

struct VSOutput
{
    float4 position : SV_POSITION;
};

struct PSOut
{
    uint triangleId : SV_Target0;
};

PSOut main(VSOutput input, uint primId : SV_PrimitiveID)
{
    PSOut o;
    o.triangleId = primId;
    return o;
}
