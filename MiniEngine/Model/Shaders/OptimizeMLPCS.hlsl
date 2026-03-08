#define FLOAT_PACK_CONSTANT 65536.0f

cbuffer OptimizerParams : register(b0) {
    float learningRate; float beta1; float beta2; float epsilon; uint totalVertices; uint totalMLPParams;
};

RWByteAddressBuffer MLPBuffer          : register(u3);
RWByteAddressBuffer MLPAdamBuffer      : register(u4);
RWByteAddressBuffer MLPGradientBuffer  : register(u5);

[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    uint pIdx = dtid.x;
    if (pIdx >= totalMLPParams) return;

    uint byteOffset = pIdx * 4;       
    uint adamByteOffset = pIdx * 8;   

    int packedGrad = MLPGradientBuffer.Load(byteOffset);
    float grad = float(packedGrad) / FLOAT_PACK_CONSTANT;

    float weight = asfloat(MLPBuffer.Load(byteOffset));
    float2 adam = asfloat(MLPAdamBuffer.Load2(adamByteOffset)); 

    adam.x = beta1 * adam.x + (1.0f - beta1) * grad;
    adam.y = beta2 * adam.y + (1.0f - beta2) * (grad * grad);

    weight -= learningRate * adam.x / (sqrt(adam.y) + epsilon);

    MLPBuffer.Store(byteOffset, asuint(weight));
    MLPAdamBuffer.Store2(adamByteOffset, asuint(adam));
    MLPGradientBuffer.Store(byteOffset, 0); 
}