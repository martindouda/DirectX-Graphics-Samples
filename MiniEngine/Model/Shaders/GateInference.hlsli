// File: GateInference.hlsli

#define LEAKY_RELU_SLOPE 0.01f

float4 activationFunction(float4 v)
{
    // Leaky ReLU
    return float4(
        (v.x >= 0.0f) ? v.x : (v.x * LEAKY_RELU_SLOPE),
        (v.y >= 0.0f) ? v.y : (v.y * LEAKY_RELU_SLOPE),
        (v.z >= 0.0f) ? v.z : (v.z * LEAKY_RELU_SLOPE),
        (v.w >= 0.0f) ? v.w : (v.w * LEAKY_RELU_SLOPE)
    );
}

float4 activationFunctionOutput(float4 v)
{
    // Sigmoid (clamps output colors nicely to 0.0 -> 1.0)
    return float4(
        1.0f / (1.0f + exp(-v.x)),
        1.0f / (1.0f + exp(-v.y)),
        1.0f / (1.0f + exp(-v.z)),
        1.0f / (1.0f + exp(-v.w))
    );
}

// Loads 4 floats at a time (1 quartet) from the raw byte buffer
float4 loadNNParameter(const uint index, inout ByteAddressBuffer nnParameters)
{
    return asfloat(nnParameters.Load4(index * 16)); // 16 bytes = 4 floats
}

// Highly optimized Fused Multiply-Add loop
void evalLayer(
    inout float4 previousActivations[MAX_NEURON_QUARTETS_PER_LAYER],
    inout float4 currentActivations[MAX_NEURON_QUARTETS_PER_LAYER],
    uint paramOffset,
    const uint neuronQuartetCountCurrentLayer,
    const uint neuronQuartetCountPreviousLayer,
    const uint layerType)
{
    for (uint neuronQuartet = 0; neuronQuartet < neuronQuartetCountCurrentLayer; neuronQuartet++)
    {
        float4 neuronValue = 0.0f;
        
        for (uint previousNeuronQuartet = 0; previousNeuronQuartet < neuronQuartetCountPreviousLayer; previousNeuronQuartet++)
        {
            const float4 prevAct = previousActivations[previousNeuronQuartet];
            neuronValue.x += dot(MLPParameterBuffer[paramOffset++], prevAct);
            neuronValue.y += dot(MLPParameterBuffer[paramOffset++], prevAct);
            neuronValue.z += dot(MLPParameterBuffer[paramOffset++], prevAct);
            neuronValue.w += dot(MLPParameterBuffer[paramOffset++], prevAct);
        }
        
        const float4 bias = MLPParameterBuffer[paramOffset++];

        if (layerType == HIDDEN_LAYER)
            currentActivations[neuronQuartet] = activationFunction(neuronValue + bias);
        else
            currentActivations[neuronQuartet] = activationFunctionOutput(neuronValue + bias);
    }
}