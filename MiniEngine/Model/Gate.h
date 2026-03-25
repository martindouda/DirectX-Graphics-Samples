// Gate.h

#pragma once

#include "GraphicsCore.h"
#include "BufferManager.h"
#include "PipelineState.h"
#include "RootSignature.h"
#include "CommandContext.h"
#include "Camera.h"
#include "ModelH3D.h"

namespace Sponza
{
    struct Int3
    {
        int32_t x, y, z;

        bool operator==(const Int3& other) const {
            return x == other.x && y == other.y && z == other.z;
        }
    };

    struct Int3Hash
    {
        std::size_t operator()(const Int3& k) const {
            // Jednoduchý prostorový hash (tzv. prime hashe)
            return ((k.x * 73856093) ^ (k.y * 19349663) ^ (k.z * 83492791));
        }
    };

    class Gate
    {
    public:
        Gate();
        ~Gate();

        // Sets up all buffers, PSOs, and initial random weights
        void Startup(const ModelH3D& model, DXGI_FORMAT colorFormat, DXGI_FORMAT depthFormat);

        // Dispatches the compute shaders to backprop and optimize
        void Train(ComputeContext& trainCtx, ColorBuffer& visibilityBuffer);

        // Renders the forward pass (inference) to a target buffer
        void RenderVisualization(GraphicsContext& gfxContext, const Math::Camera& camera, DepthBuffer& depthBuffer,
            const D3D12_VIEWPORT& viewport, const D3D12_RECT& scissor, ColorBuffer& visibilityBuffer);

        // Draws the ImGui interface
        void RenderGUI();

        // Resets the neural network and Adam states back to step 1
        void ResetTraining();


        void Cleanup();

        inline ColorBuffer& GetGateColorBuffer() { return m_GateColorBuffer; }
        inline ColorBuffer& GetVisColorBuffer() { return m_VisColorBuffer; }
        inline void SetIsTrainingPaused(bool isTrainingPaused) { m_IsTrainingPaused = isTrainingPaused; }
        inline bool GetIsTrainingPaused() { return m_IsTrainingPaused; }

    private:
        struct GlobalTriangle
        {
            uint32_t i0, i1, i2;
            uint32_t materialIdx;
        };

        struct GateFeature
        {
            DirectX::XMFLOAT4 data[2];
        };

        struct AdamData
        {
            DirectX::XMFLOAT4 mean;
            DirectX::XMFLOAT4 variance;
            uint32_t stepCount;
            uint32_t pad[3];
        };

        const ModelH3D* m_Model;

        // --- Network State & Buffers ---
        uint32_t m_TotalVertices = 0;
        uint32_t m_TotalTriangles = 0;

        uint32_t m_UniqueSpatialVertexCount = 0;

        StructuredBuffer m_GateFeatureBuffer;
        ByteAddressBuffer m_GateFeatureGradientBuffer;
        ByteAddressBuffer m_GateFeatureAdamBuffer;

        ByteAddressBuffer m_GateMLPBuffer;
        ByteAddressBuffer m_GateMLPGradientBuffer;
        ByteAddressBuffer m_GateMLPAdamBuffer;

        StructuredBuffer m_GlobalTriangleBuffer;
        StructuredBuffer m_VertexMaterialMap;

        StructuredBuffer m_SpatialTriangleBuffer;

        // PØIDÁNO: MESH COLORS BUFFERY
        StructuredBuffer m_UniqueFeatureBuffer;
        StructuredBuffer m_VertexMappingBuffer;

        // --- PSOs and Root Signatures ---
        // Inference
        GraphicsPSO m_GatePSO;
        RootSignature m_GateRootSig;
        ColorBuffer m_GateColorBuffer;

        // Training
        RootSignature m_GateTrainRootSig;
        ComputePSO m_GateBackpropPSO;
        ComputePSO m_GateOptMLPPSO;
        ComputePSO m_GateOptFeatPSO;

        // PØIDÁNO: BROADCAST PSO
        ComputePSO m_GateBroadcastPSO;

        // Utils
        RootSignature m_EncodeColorRootSig;
        ComputePSO m_EncodeColorPSO;

        // --- Hyperparameters & Training State ---
        uint32_t m_TrainingStep = 1;

        float m_FeatureLearningRate = 0.05f;
        float m_MLPLearningRate = 0.002f;
        float m_AdamEpsilon = 1e-8f;
        float m_AdamBeta1 = 0.9f;
        float m_AdamBeta2 = 0.999f;
        float m_WeightDecay = 0.01f;

        float m_ScreenSpaceRatio = 0.85f;

        bool m_IsTrainingPaused = true;
        int m_BackpropDispatchedGroups = 8192; // * 64 triangles per step

        // Custom parameters for debugging and experimentation
        int m_CustomInt0 = 0;


        ColorBuffer m_VisColorBuffer;
        ComputePSO m_VisPSO;
        RootSignature m_VisRootSig;
    };
}