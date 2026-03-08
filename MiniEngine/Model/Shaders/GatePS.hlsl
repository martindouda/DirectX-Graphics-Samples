// GatePS.hlsl

// --- Architecture Macros (8 -> 16 -> 4) ---
#define MAX_NEURON_QUARTETS_PER_LAYER 4 // Max 16 neurons / 4
#define HIDDEN_LAYER 1
#define OUTPUT_LAYER 2

ByteAddressBuffer nnParametersInputBuffer : register(t1);

#include "GateInference.hlsli"

struct VSOutput 
{
    float4 Position : SV_POSITION;
    float4 f0 : TEXCOORD0;
    float4 f1 : TEXCOORD1;
};


float4 main(VSOutput input) : SV_TARGET
{
    float4 activationsA[MAX_NEURON_QUARTETS_PER_LAYER];
    float4 activationsB[MAX_NEURON_QUARTETS_PER_LAYER];

    // 1. Load the interpolated features
    activationsA[0] = input.f0;
    activationsA[1] = input.f1;
    activationsA[2] = 0.0f; // Padding for 16-neuron max size
    activationsA[3] = 0.0f;

    // 2. LAYER 1 (8 Inputs -> 16 Hidden)
    // Current Quartets: 4 (16 neurons). Previous: 2 (8 inputs).
    // Offset: 0. 
    // This layer consumes 36 quartets total (32 for weights + 4 for biases).
    evalLayer(activationsA, activationsB, nnParametersInputBuffer, 0, 4, 2, HIDDEN_LAYER);

    // 3. LAYER 2 (16 Hidden -> 4 Outputs)
    // Current Quartets: 1 (4 outputs). Previous: 4 (16 hidden).
    // Offset: Starts at 36.
    evalLayer(activationsB, activationsA, nnParametersInputBuffer, 36, 1, 4, OUTPUT_LAYER);

    // 4. Return Output (Color is stored in the first quartet of the output array)
    return float4(activationsA[0].xyz, 1.0f);
    //return input.f0;
}