#define FLOAT_PACK_CONSTANT 65536.0f

cbuffer OptimizerParams : register(b0) {
    float learningRate; float beta1; float beta2; float epsilon; uint totalVertices; uint totalMLPParams;
};

struct GateFeature { float4 data[2]; };
struct AdamState   { float4 m[2]; float4 v[2]; };

RWStructuredBuffer<GateFeature> FeatureBuffer      : register(u0);
RWStructuredBuffer<AdamState>   FeatureAdamBuffer  : register(u1);
RWByteAddressBuffer             FeatureGradients   : register(u2);

[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    uint vIdx = dtid.x;
    if (vIdx >= totalVertices) return;

    GateFeature feat = FeatureBuffer[vIdx];
    AdamState adam = FeatureAdamBuffer[vIdx];
    uint byteOffset = vIdx * 32; 

    [unroll]
    for (int i = 0; i < 2; ++i)
    {
        int4 packedGrad = FeatureGradients.Load4(byteOffset + i * 16);
        float4 grad = float4(packedGrad) / FLOAT_PACK_CONSTANT;

        adam.m[i] = beta1 * adam.m[i] + (1.0f - beta1) * grad;
        adam.v[i] = beta2 * adam.v[i] + (1.0f - beta2) * (grad * grad);

        feat.data[i] -= learningRate * adam.m[i] / (sqrt(adam.v[i]) + epsilon);

        FeatureGradients.Store4(byteOffset + i * 16, int4(0, 0, 0, 0));
    }

    FeatureBuffer[vIdx] = feat;
    FeatureAdamBuffer[vIdx] = adam;
}