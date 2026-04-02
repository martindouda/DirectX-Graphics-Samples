// Gate.h

#pragma once

#include "GraphicsCore.h"
#include "BufferManager.h"
#include "ReadbackBuffer.h"
#include "PipelineState.h"
#include "RootSignature.h"
#include "CommandContext.h"
#include "Camera.h"
#include "ModelH3D.h"

namespace Sponza
{
    // --- Spatial hashing ---
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
            // Simple spatial hash
            return ((k.x * 73856093) ^ (k.y * 19349663) ^ (k.z * 83492791));
        }
    };

    class Gate
    {
    public:
        Gate();
        ~Gate();

        // --- Main API ---
        void Startup(const ModelH3D& model, DXGI_FORMAT colorFormat, DXGI_FORMAT depthFormat);
        void Train(ComputeContext& trainCtx, ColorBuffer& visibilityBuffer, Math::Vector3 sunDirection);
        void RenderVisualization(GraphicsContext& gfxContext, const Math::Camera& camera, DepthBuffer& depthBuffer,
            const D3D12_VIEWPORT& viewport, const D3D12_RECT& scissor, ColorBuffer& visibilityBuffer,
            Math::Vector3 sunDirection, float sunIntensity);

        void RenderGUI();
        void ResetTraining();
        void Cleanup();

        inline ColorBuffer& GetGateColorBuffer() { return m_GateColorBuffer; }
        inline ColorBuffer& GetVisColorBuffer() { return m_VisColorBuffer; }
        inline void SetIsTrainingPaused(bool isTrainingPaused) { m_IsTrainingPaused = isTrainingPaused; }
        inline bool GetIsTrainingPaused() const { return m_IsTrainingPaused; }

    private:
        // --- Init helpers ---
        void BuildSpatialIndex();
        void AllocateBuffers();
        void InitializePSOs(DXGI_FORMAT colorFormat, DXGI_FORMAT depthFormat);

        // --- GPU data structures ---
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

        // --- Geometry ---
        uint32_t m_TotalVertices = 0;
        uint32_t m_TotalTriangles = 0;
        uint32_t m_UniqueSpatialVertexCount = 0;

        // Geometry buffers
        StructuredBuffer m_GlobalTriangleBuffer;
        StructuredBuffer m_VertexMaterialMap;

        // Feature buffers (Cache-coherency architecture)
        StructuredBuffer m_GateFeatureBuffer;           // Duplicated   (for fast read during Inference/Backprop)
        StructuredBuffer m_UniqueFeatureBuffer;         // Unique       (for write by Adam optimizer)
        StructuredBuffer m_VertexMappingBuffer;         // N:M mapping  (for copying data to duplicates)

        ByteAddressBuffer m_GateFeatureGradientBuffer;
        ByteAddressBuffer m_GateFeatureAdamBuffer;

        // MLP buffers
        ByteAddressBuffer m_GateMLPBuffer;
        ByteAddressBuffer m_GateMLPGradientBuffer;
        ByteAddressBuffer m_GateMLPAdamBuffer;

        // --- PSOs a Root Signatures ---
        // Inference
        RootSignature m_GateRootSig;
        GraphicsPSO m_GatePSO;
        ColorBuffer m_GateColorBuffer;

        // Training
        RootSignature m_GateTrainRootSig;
        ComputePSO m_GateBackpropPSO;
        ComputePSO m_GateOptMLPPSO;
        ComputePSO m_GateOptFeatPSO;
        ComputePSO m_GateBroadcastPSO;

        // Vis
        RootSignature m_VisRootSig;
        ComputePSO m_VisPSO;
        ColorBuffer m_VisColorBuffer;

        RootSignature m_EncodeColorRootSig;
        ComputePSO m_EncodeColorPSO;

		// --- Hyperparameters and state ---
        uint32_t m_Resolution = 4;
        int m_DesiredResolution = 4; // For UI
        uint32_t m_PointsPerTri;

        uint32_t m_TrainingStep = 1;
        bool m_IsTrainingPaused = true;

        int m_BackpropDispatchedGroups = 1024; // * 1024 trojúhelníkù na krok
        float m_GlobalLearningRate = 0.1f;
        float m_LearningRateRatio = 0.5f;
        float m_AdamEpsilon = 1e-8f;
        float m_AdamBeta1 = 0.9f;
        float m_AdamBeta2 = 0.999f;
        float m_WeightDecay = 0.01f;
        float m_ScreenSpaceRatio = 0.85f;

        // Pøidej nìkam k promìnným tøídy Gate:
        std::vector<float> m_LossHistory;
        uint32_t m_LossHistoryOffset = 0;
        const uint32_t MAX_LOSS_HISTORY = 100;

		ByteAddressBuffer m_LossBuffer;        // GPU writes here the loss value after each training step
		ReadbackBuffer m_LossReadbackBuffer;   // CPU reads the loss value from here

        Microsoft::WRL::ComPtr<ID3D12QueryHeap> m_GpuTimerHeap;
        Microsoft::WRL::ComPtr<ID3D12Resource> m_GpuTimerReadback;
        uint64_t m_GpuTimestampFreq = 0;

        // --- Promìnné pro ImGui ---
        float m_GpuTimeBackprop = 0.0f;
        float m_GpuTimeOptMLP = 0.0f;
        float m_GpuTimeOptFeat = 0.0f;
        float m_GpuTimeBroadcast = 0.0f;
        float m_GpuTimeRender = 0.0f;
        float m_CpuTimeBuildSpatialIndex = 0.0f;
    };
}