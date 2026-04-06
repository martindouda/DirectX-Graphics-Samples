// File: GatePS.hlsl

#define GATE_INFERENCE
#include "GateTrainCommon.hlsli"

// =========================================================================
//  Constant Buffers & Structures
// =========================================================================

cbuffer MeshConstants : register(b1) 
{ 
    uint globalTriangleOffset; 
    uint lightingMode;
    uint renderFlags;
    uint materialIdx;

    float3 sunDirection;
    float sunIntensity;
};

struct VSOutput 
{
    float4 Position  : SV_POSITION; 
    float2 UV        : TEXCOORD0; // Ensure the Vertex Shader outputs UVs
    float3 Normal    : NORMAL;
    float3 Tangent   : TANGENT;
    float3 Bitangent : BITANGENT;
    float3 worldPos  : WorldPos;
};

// =========================================================================
//  PIXEL SHADER: Forward Inference & Lighting
// =========================================================================

float3 ComputeNormal(VSOutput input)
{
    // 1. Sample and Unpack the Normal Map
    // (Ensure register t3 is your normal map)
    float3 mapNormal = BindlessTextures[materialIdx * 6 + 3].Sample(LinearSampler, input.UV).rgb;
    mapNormal = mapNormal * 2.0 - 1.0;

    // 2. Construct the TBN matrix exactly like your provided Pixel Shader
    // We normalize the interpolated vectors to ensure the basis is orthonormal
    float3x3 tbn = float3x3(normalize(input.Tangent), normalize(input.Bitangent), normalize(input.Normal));

    // 3. Transform Tangent Space -> World Space
    // mul(vector, matrix) in HLSL performs a Row-Vector * Matrix multiplication
    return normalize(mul(mapNormal, tbn));
}

float4 main(VSOutput input, uint primitiveID : SV_PrimitiveID, float3 barycentrics : SV_Barycentrics) : SV_TARGET
{
    // --- 1. NEURAL NETWORK INFERENCE ---
    uint globalTriID = primitiveID + globalTriangleOffset;
    
    // 1. Fetch properties directly from the Triangle Buffer
    GlobalTriangle triData = GlobalTriangleBuffer[globalTriID];
    uint baseIndex = triData.pointOffset;
    uint localRes = triData.resolution;

    uint i0, j0, i1, j1, i2, j2;
    float weight0, weight1, weight2;
    
    // Pass localRes instead of meshColorResolution
    getMeshColorIndicesAndWeights(barycentrics, localRes, i0, j0, weight0, i1, j1, weight1, i2, j2, weight2);

    uint idx0 = get1DIndex(i0, j0, localRes);
    uint idx1 = get1DIndex(i1, j1, localRes);
    uint idx2 = get1DIndex(i2, j2, localRes);

    GateFeature f0 = FeatureBuffer[baseIndex + idx0];
    GateFeature f1 = FeatureBuffer[baseIndex + idx1];
    GateFeature f2 = FeatureBuffer[baseIndex + idx2];

    // Interpolate features using barycentric weights
    float4 interpF0 = weight0 * f0.data[0] + weight1 * f1.data[0] + weight2 * f2.data[0];
    float4 interpF1 = weight0 * f0.data[1] + weight1 * f1.data[1] + weight2 * f2.data[1];

    float4 activationsA[MAX_NEURON_QUARTETS_PER_LAYER];
    float4 activationsB[MAX_NEURON_QUARTETS_PER_LAYER];
    
    // Load input layer
    activationsA[0] = interpF0; 
    activationsA[1] = interpF1; 
    activationsA[2] = 0.0f; 
    activationsA[3] = 0.0f;

    // Evaluate Network (Hidden -> Output)
    evalLayer(activationsA, activationsB, 0,  4, 2, HIDDEN_LAYER);
    evalLayer(activationsB, activationsA, 36, 1, 4, OUTPUT_LAYER);

    // --- Extract both masks from the network's float4 output ---
    float networkShadow = saturate(activationsA[0].x); 
    float networkAO     = saturate(activationsA[0].y); 
    
    // Default to fully unoccluded (1.0)
    float shadowMask = 1.0f;
    float aoMask = 1.0f;

    // Apply masks based on the UI toggle
    // Mode 0: None (Masks remain 1.0)
    // Mode 1: AO Only
    // Mode 2: Shadows Only
    // Mode 3: Both
    
    if (lightingMode == 1 || lightingMode == 3)
        aoMask = networkAO;
        
    if (lightingMode == 2 || lightingMode == 3)
        shadowMask = networkShadow;
    
    // --- 2. TEXTURING & NORMAL MAPPING ---
    
    // Unpack our toggle flags
    bool useTexturelessView      = (renderFlags & (1 << 0)) != 0;
    bool disableDirectionalLight = (renderFlags & (1 << 1)) != 0;

    float4 albedo = BindlessTextures[materialIdx * 6 + 0].Sample(LinearSampler, input.UV);
    
    // Override the RGB color with a neutral clay gray if the toggle is active
    if (useTexturelessView)
    {
        albedo.rgb = float3(0.8f, 0.8f, 0.8f);
    }
    
    float3 N = ComputeNormal(input);
    float3 L = normalize(sunDirection);
    float NdotL = saturate(dot(N, L));
    
    // Direct sunlight is zeroed out if the toggle is active
    float3 directLight = 0.0f;
    if (!disableDirectionalLight)
        directLight = albedo.rgb * NdotL * sunIntensity * shadowMask; 
    
    // If we turned off the sun, boost the ambient light so we can see the AO clearly!
    float ambientBase = disableDirectionalLight ? 1.0f : 0.4f;
    float3 ambientLight = albedo.rgb * ambientBase * aoMask; 
    
    return float4(directLight + ambientLight, albedo.a);
}