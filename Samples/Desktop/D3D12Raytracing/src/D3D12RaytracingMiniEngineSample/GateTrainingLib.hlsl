//=========================================================================
// GateTrainingLib.hlsl
// DXR Stochastic Training Pass for Geometry-Aware Trained Encoding (GATE)
//=========================================================================

#define HLSL
#include "ModelViewerRaytracing.h"
#include "RayTracingHlslCompat.h" // For RayTraceMeshInfo

// --- GATE ARCHITECTURE MACROS ---
#define MAX_NEURON_QUARTETS_PER_LAYER 4 
#define HIDDEN_LAYER 1
#define OUTPUT_LAYER 2
#define FLOAT_PACK_CONSTANT 65536.0f

// --- GATE RW BUFFERS (Global Root Signature Slots u12 - u15) ---
struct GateFeature { float4 data[2]; }; 
RWStructuredBuffer<GateFeature> FeatureBuffer      : register(u12);
RWByteAddressBuffer             FeatureGradients   : register(u13);
RWByteAddressBuffer             MLPBuffer          : register(u14);
RWByteAddressBuffer             MLPGradientBuffer  : register(u15);

// --- SCENE BUFFERS (Global Root Signature Slots) ---
StructuredBuffer<RayTraceMeshInfo> g_meshInfo      : register(t1);
ByteAddressBuffer                  g_indices       : register(t2);
ByteAddressBuffer                  g_attributes    : register(t3);

// --- MATERIAL TEXTURE (Local Root Signature Slots) ---
Texture2D<float4>                  g_localTexture  : register(t6);
SamplerState                       g_s0            : register(s0);

// --- HELPER FUNCTIONS ---

// Atomic addition for floats packed into integers
void AtomicAddFloat(RWByteAddressBuffer buf, uint byteAddress, float val)
{
    int packed = int(val * FLOAT_PACK_CONSTANT);
    buf.InterlockedAdd(byteAddress, packed);
}

// PCG Hash for Random Number Generation
uint pcgHash(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float rand(inout uint rngState) {
    rngState = pcgHash(rngState);
    return asfloat(0x3f800000 | (rngState >> 9)) - 1.0f;
}

// Loads 3 16-bit indices from the MiniEngine Index Buffer
uint3 Load3x16BitIndices(uint offsetBytes)
{
    const uint dwordAlignedOffset = offsetBytes & ~3;
    const uint2 four16BitIndices = g_indices.Load2(dwordAlignedOffset);
    uint3 indices;

    if (dwordAlignedOffset == offsetBytes) {
        indices.x = four16BitIndices.x & 0xffff;
        indices.y = (four16BitIndices.x >> 16) & 0xffff;
        indices.z = four16BitIndices.y & 0xffff;
    } else {
        indices.x = (four16BitIndices.x >> 16) & 0xffff;
        indices.y = four16BitIndices.y & 0xffff;
        indices.z = (four16BitIndices.y >> 16) & 0xffff;
    }
    return indices;
}

float2 GetUVAttribute(uint byteOffset) {
    return asfloat(g_attributes.Load2(byteOffset));
}

// Include MLPZen Math
#define MLP_BUFFER_TYPE RWByteAddressBuffer
#include "GateInference.hlsli"

// ============================================================================
// 1. STOCHASTIC RAY GENERATION
// ============================================================================
[shader("raygeneration")]
void RayGen()
{
    // Seed RNG based on Thread ID + FrameCount (Assume FrameCount is in g_dynamic.padding for now)
    uint2 dispatchIdx = DispatchRaysIndex().xy;
    uint rngState = pcgHash(dispatchIdx.x + dispatchIdx.y * 256 + g_dynamic.padding * 65536);

    // Pick a random pixel
    float2 randomScreenPos = float2(rand(rngState), rand(rngState));
    float2 ndc = randomScreenPos * 2.0f - 1.0f;
    ndc.y = -ndc.y;

    // Unproject to Ray
    float4 unprojected = mul(g_dynamic.cameraToWorld, float4(ndc, 0, 1));
    float3 world = unprojected.xyz / unprojected.w;
    
    RayDesc ray;
    ray.Origin = g_dynamic.worldCameraPosition;
    ray.Direction = normalize(world - ray.Origin);
    ray.TMin = 0.0f;
    ray.TMax = FLT_MAX;

    RayPayload payload;
    payload.SkipShading = false;
    payload.RayHitT = FLT_MAX;
    
    TraceRay(g_accel, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, ~0, 0, 1, 0, ray, payload);
}

[shader("miss")]
void Miss(inout RayPayload payload)
{
    payload.SkipShading = true;
}


// Scatter 4 floats at once to keep the code clean
void AtomicAddFloat4(RWByteAddressBuffer buf, uint baseAddress, float4 val) 
{
    AtomicAddFloat(buf, baseAddress + 0,  val.x);
    AtomicAddFloat(buf, baseAddress + 4,  val.y);
    AtomicAddFloat(buf, baseAddress + 8,  val.z);
    AtomicAddFloat(buf, baseAddress + 12, val.w);
}

// Derivative of the Leaky ReLU activation function
float4 leakyReluDeriv(float4 act) 
{
    // If activation is > 0, gradient passes through (1.0). Else, scaled by slope (0.01).
    return float4(
        act.x > 0.0f ? 1.0f : 0.01f,
        act.y > 0.0f ? 1.0f : 0.01f,
        act.z > 0.0f ? 1.0f : 0.01f,
        act.w > 0.0f ? 1.0f : 0.01f
    );
}


// ============================================================================
// 2. CLOSEST HIT (FORWARD & BACKWARD PASS)
// ============================================================================
[shader("closesthit")]
void Hit(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    payload.RayHitT = RayTCurrent();

    // --------------------------------------------------------
    // A. FETCH GEOMETRY & BARYCENTRICS
    // --------------------------------------------------------
    float u = attr.barycentrics.x;
    float v = attr.barycentrics.y;
    float w = 1.0f - u - v;

    RayTraceMeshInfo info = g_meshInfo[InstanceID()]; 

    // Get Vertices
    uint3 ii = Load3x16BitIndices(info.m_indexOffsetBytes + PrimitiveIndex() * 3 * 2);
    
    // Add base vertex offset!
    uint baseVertex = info.m_positionAttributeOffsetBytes / info.m_attributeStrideBytes;
    uint v0 = baseVertex + ii.x;
    uint v1 = baseVertex + ii.y;
    uint v2 = baseVertex + ii.z;

    // Get UVs
    float2 uv0 = GetUVAttribute(info.m_uvAttributeOffsetBytes + ii.x * info.m_attributeStrideBytes);
    float2 uv1 = GetUVAttribute(info.m_uvAttributeOffsetBytes + ii.y * info.m_attributeStrideBytes);
    float2 uv2 = GetUVAttribute(info.m_uvAttributeOffsetBytes + ii.z * info.m_attributeStrideBytes);
    float2 interpUV = w * uv0 + u * uv1 + v * uv2;

    // --------------------------------------------------------
    // B. FETCH & INTERPOLATE NEURAL FEATURES
    // --------------------------------------------------------
    GateFeature f0 = FeatureBuffer[v0];
    GateFeature f1 = FeatureBuffer[v1];
    GateFeature f2 = FeatureBuffer[v2];

    float4 interpF0 = f0.data[0] * w + f1.data[0] * u + f2.data[0] * v;
    float4 interpF1 = f0.data[1] * w + f1.data[1] * u + f2.data[1] * v;

    // --------------------------------------------------------
    // C. FORWARD PASS (MLP)
    // --------------------------------------------------------
    float4 activationsA[MAX_NEURON_QUARTETS_PER_LAYER];
    float4 activationsB[MAX_NEURON_QUARTETS_PER_LAYER];
    
    activationsA[0] = interpF0;
    activationsA[1] = interpF1;
    activationsA[2] = 0.0f; 
    activationsA[3] = 0.0f;

    // Layer 1 (Hidden): activationsA -> activationsB
    evalLayer(activationsA, activationsB, MLPBuffer, 0, 4, 2, HIDDEN_LAYER);
    // Layer 2 (Output): activationsB -> activationsA
    evalLayer(activationsB, activationsA, MLPBuffer, 36, 1, 4, OUTPUT_LAYER);

    float4 predictedColor = activationsA[0];

    // --------------------------------------------------------
    // D. CALCULATE LOSS
    // --------------------------------------------------------
    float4 groundTruth = g_localTexture.SampleLevel(g_s0, interpUV, 0);

    // Derivative of L2 Loss = 2 * (Predicted - GroundTruth)
    // We scale it down slightly (e.g., 0.1) as a localized learning rate damper
    float4 dLoss_dOutput = 0.1f * 2.0f * (predictedColor - groundTruth);

    // --------------------------------------------------------
    // E. BACKWARD PASS (MLPZen Unrolled Matrix Multiplications)
    // --------------------------------------------------------
    
    // --- LAYER 2 BACKPROP ---
    // Sigmoid Derivative: a * (1 - a)
    float4 delta2 = dLoss_dOutput * (predictedColor * (1.0f - predictedColor));
    
    // Prepare to catch errors flowing back to Layer 1
    float4 error1[4] = { float4(0,0,0,0), float4(0,0,0,0), float4(0,0,0,0), float4(0,0,0,0) };

    uint paramOffset = 36;
    for (uint hq = 0; hq < 4; hq++) // Loop over the 4 hidden quartets (16 neurons)
    {
        float4 prevAct = activationsB[hq];
        
        // Output X neuron weights
        float4 wX = loadNNParameter(paramOffset, MLPBuffer);
        AtomicAddFloat4(MLPGradientBuffer, paramOffset * 16, delta2.x * prevAct);
        error1[hq] += wX * delta2.x;
        paramOffset++;
        
        // Output Y neuron weights
        float4 wY = loadNNParameter(paramOffset, MLPBuffer);
        AtomicAddFloat4(MLPGradientBuffer, paramOffset * 16, delta2.y * prevAct);
        error1[hq] += wY * delta2.y;
        paramOffset++;
        
        // Output Z neuron weights
        float4 wZ = loadNNParameter(paramOffset, MLPBuffer);
        AtomicAddFloat4(MLPGradientBuffer, paramOffset * 16, delta2.z * prevAct);
        error1[hq] += wZ * delta2.z;
        paramOffset++;
        
        // Output W neuron weights
        float4 wW = loadNNParameter(paramOffset, MLPBuffer);
        AtomicAddFloat4(MLPGradientBuffer, paramOffset * 16, delta2.w * prevAct);
        error1[hq] += wW * delta2.w;
        paramOffset++;
    }
    // Layer 2 Bias Gradients
    AtomicAddFloat4(MLPGradientBuffer, paramOffset * 16, delta2);
    paramOffset++;


    // --- LAYER 1 BACKPROP ---
    float4 delta1[4];
    for (uint i = 0; i < 4; i++) {
        delta1[i] = error1[i] * leakyReluDeriv(activationsB[i]);
    }

    // Catch errors flowing back to the Input Features
    float4 error0[2] = { float4(0,0,0,0), float4(0,0,0,0) }; 

    paramOffset = 0;
    for (uint currQ = 0; currQ < 4; currQ++) // Loop over 4 hidden quartets
    {
        for (uint prevQ = 0; prevQ < 2; prevQ++) // Loop over 2 input quartets
        {
            float4 prevAct = activationsA[prevQ]; // Original features
            
            float4 wX = loadNNParameter(paramOffset, MLPBuffer);
            AtomicAddFloat4(MLPGradientBuffer, paramOffset * 16, delta1[currQ].x * prevAct);
            error0[prevQ] += wX * delta1[currQ].x;
            paramOffset++;
            
            float4 wY = loadNNParameter(paramOffset, MLPBuffer);
            AtomicAddFloat4(MLPGradientBuffer, paramOffset * 16, delta1[currQ].y * prevAct);
            error0[prevQ] += wY * delta1[currQ].y;
            paramOffset++;
            
            float4 wZ = loadNNParameter(paramOffset, MLPBuffer);
            AtomicAddFloat4(MLPGradientBuffer, paramOffset * 16, delta1[currQ].z * prevAct);
            error0[prevQ] += wZ * delta1[currQ].z;
            paramOffset++;
            
            float4 wW = loadNNParameter(paramOffset, MLPBuffer);
            AtomicAddFloat4(MLPGradientBuffer, paramOffset * 16, delta1[currQ].w * prevAct);
            error0[prevQ] += wW * delta1[currQ].w;
            paramOffset++;
        }
        // Layer 1 Bias Gradients
        AtomicAddFloat4(MLPGradientBuffer, paramOffset * 16, delta1[currQ]);
        paramOffset++;
    }

    // --------------------------------------------------------
    // F. SCATTER GRADIENTS TO VERTICES
    // --------------------------------------------------------
    float4 inputGradient0 = error0[0]; 
    float4 inputGradient1 = error0[1];

    // Scatter to V0 (Weight = w)
    uint byteAddr0 = v0 * 32; 
    AtomicAddFloat4(FeatureGradients, byteAddr0 + 0,  inputGradient0 * w);
    AtomicAddFloat4(FeatureGradients, byteAddr0 + 16, inputGradient1 * w);

    // Scatter to V1 (Weight = u)
    uint byteAddr1 = v1 * 32;
    AtomicAddFloat4(FeatureGradients, byteAddr1 + 0,  inputGradient0 * u);
    AtomicAddFloat4(FeatureGradients, byteAddr1 + 16, inputGradient1 * u);

    // Scatter to V2 (Weight = v)
    uint byteAddr2 = v2 * 32;
    AtomicAddFloat4(FeatureGradients, byteAddr2 + 0,  inputGradient0 * v);
    AtomicAddFloat4(FeatureGradients, byteAddr2 + 16, inputGradient1 * v);
}