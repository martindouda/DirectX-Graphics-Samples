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

#include <vector>
#include <cstdint>

namespace Sponza
{
    // --- Spatial Hashing ---
    struct Int3
    {
        int32_t x, y, z;

        bool operator==(const Int3& other) const
        {
            return x == other.x && y == other.y && z == other.z;
        }
    };

    struct Int3Hash
    {
        std::size_t operator()(const Int3& k) const
        {
            // Simple spatial hash
            return ((k.x * 73856093) ^ (k.y * 19349663) ^ (k.z * 83492791));
        }
    };

    class Gate
    {
    public:
        Gate();
        ~Gate();

        // --- Lifecycle ---
        void Startup(const ModelH3D& model, DXGI_FORMAT colorFormat, DXGI_FORMAT depthFormat);
        void Cleanup();

        // --- Execution ---
        void Train(ComputeContext& trainCtx, ColorBuffer& visibilityBuffer, Math::Vector3 sunDirection);
        void RenderVisualization(GraphicsContext& gfxContext, const Math::Camera& camera, DepthBuffer& depthBuffer,
            const D3D12_VIEWPORT& viewport, const D3D12_RECT& scissor, ColorBuffer& visibilityBuffer,
            Math::Vector3 sunDirection, float sunIntensity);
        void RenderGUI();
        void ResetTraining();

        // --- Accessors ---
        inline ColorBuffer& GetGateColorBuffer() { return m_GateColorBuffer; }
        inline ColorBuffer& GetVisColorBuffer() { return m_VisColorBuffer; }

        inline void SetIsTrainingPaused(bool isTrainingPaused) { m_IsTrainingPaused = isTrainingPaused; }
        inline bool GetIsTrainingPaused() const { return m_IsTrainingPaused; }
		inline void SetTexturesEnabled(bool enabled) { m_TexturesEnabled = enabled; }
		inline bool GetTexturesEnabled() const { return m_TexturesEnabled; }
    private:
        // --- Initialization Helpers ---
        void BuildSpatialIndex();
        void AllocateBuffers();
        void InitializePSOs(DXGI_FORMAT colorFormat, DXGI_FORMAT depthFormat);

        // --- GPU Data Structures ---
        struct GlobalTriangle
        {
            uint32_t i0, i1, i2, materialIdx;
            uint32_t pointOffset;  // The starting index in the flat point arrays
            uint32_t resolution;   // The specific resolution for this triangle
            uint32_t pointsPerTri; // The number of points for this triangle
            uint32_t pad;
        };

        struct AdamData
        {
            DirectX::XMFLOAT4 mean;
            DirectX::XMFLOAT4 variance;
            uint32_t stepCount;
            uint32_t pad[3];
        };

        // --- Core State ---
        const ModelH3D* m_Model = nullptr;

        // --- Geometry Statistics ---
        uint32_t m_TotalVertices = 0;
        uint32_t m_TotalTriangles = 0;
        uint32_t m_TotalMeshColorPoints = 0;
        uint32_t m_UniqueSpatialVertexCount = 0;

        // --- Hyperparameters & Training Configuration ---
        bool     m_IsTrainingPaused = true;
        uint32_t m_TrainingStep = 1;
        uint32_t m_Resolution = 8;
        int      m_DesiredResolution = 0;        // For UI
        uint32_t m_PointsPerTri = 0;
        bool     m_UseMaxEdgeLength = false;
        bool     m_DesiredUseMaxEdgeLength = false;
        int      m_LearningMode = 0;

        // Add to your Hyperparameters section:
        uint32_t m_FeatureFloats = 4;
        uint32_t m_DesiredFeatureFloats = 0; // For UI
        uint32_t m_FeatureQuartets = 1;

        bool m_UseDeduplication = true;
        bool m_DesiredUseDeduplication = true;

        // Add to track dynamic MLP size:
        uint32_t m_MlpParameterCount = 212;
        uint32_t m_MlpQuartets = 53;

        int      m_BackpropDispatchedGroups = 1024; // * 1024 triangles per step
        float    m_GlobalLearningRate = 0.1f;
        float    m_LearningRateRatio = 0.5f;
        float    m_MaxGradientClip = 1.0f;
        float    m_AdamEpsilon = 1e-8f;
        float    m_AdamBeta1 = 0.9f;
        float    m_AdamBeta2 = 0.999f;
        float    m_WeightDecay = 0.01f;
        float    m_ScreenSpaceRatio = 0.85f;
        float    m_AoRadius = 150.0f;
        int      m_LightingMode = 3; // 0 = None, 1 = AO Only, 2 = Shadows Only, 3 = Both
        bool     m_TexturesEnabled = true;
        bool     m_DirectionalLightEnabled = true;

        // --- GPU Resources: Geometry & Features ---
        StructuredBuffer m_GlobalTriangleBuffer;          // Triangle metadata (indices, resolution, point offsets)

        // Cache-coherency architecture buffers
        StructuredBuffer m_GateFeatureBuffer;             // Duplicated features for fast O(1) read during Inference
        StructuredBuffer m_UniqueFeatureBuffer;           // Unique features for safe atomic writes during Backprop
		StructuredBuffer m_VertexMappingBuffer;           // Maps linear point indices to unique feature IDs, used for gradient accumulation in backrop

        StructuredBuffer m_GateFeatureGradientBuffer;     // Accumulated loss gradients for unique features
        StructuredBuffer m_GateFeatureAdamBuffer;         // AdamW optimizer state (mean, variance) for features

        // --- GPU Resources: MLP ---
        StructuredBuffer m_GateMLPBuffer;                 // Trainable weights and biases of the MLP network
        StructuredBuffer m_GateMLPGradientBuffer;         // Accumulated loss gradients for MLP weights
        StructuredBuffer m_GateMLPAdamBuffer;             // AdamW optimizer state for MLP weights

        // --- GPU Resources: Sparse "Push" Broadcast Architecture ---
        StructuredBuffer m_GateFeatureDirtyBuffer;        // Flags indicating which unique features were updated this frame
        StructuredBuffer m_UniqueToDuplicateOffsetBuffer; // Inverted index: start offset for duplicates
        StructuredBuffer m_UniqueToDuplicateCountBuffer;  // Inverted index: total duplicates per unique feature
        StructuredBuffer m_DuplicateIndicesBuffer;        // Inverted index: flat array of all duplicate linear indices

        // --- Root Signatures & Pipeline States ---
        // Inference Pipeline
        RootSignature     m_GateRootSig;
        GraphicsPSO       m_GatePSO;
        ColorBuffer       m_GateColorBuffer;

        // Training Pipeline
        RootSignature     m_GateTrainRootSig;
        ComputePSO        m_GateBackpropPSO;
        ComputePSO        m_GateOptMLPPSO;
        ComputePSO        m_GateOptFeatPSO;
        ComputePSO        m_GateBroadcastPSO;

        // Visualization Pipeline
        RootSignature     m_VisRootSig;
        ComputePSO        m_VisPSO;
        ColorBuffer       m_VisColorBuffer;

        // Encoding Pipeline
        RootSignature     m_EncodeColorRootSig;
        ComputePSO        m_EncodeColorPSO;

        // --- Training Metrics (Loss Tracking) ---
        static const uint32_t MAX_LOSS_HISTORY = 100;
        std::vector<float>    m_LossHistory;
        uint32_t              m_LossHistoryOffset = 0;

        ByteAddressBuffer     m_LossBuffer;         // GPU writes here the loss value after each training step
        ReadbackBuffer        m_LossReadbackBuffer; // CPU reads the loss value from here

        // --- Hardware Timers & GUI Profiling Variables ---
        Microsoft::WRL::ComPtr<ID3D12QueryHeap> m_GpuTimerHeap;
        Microsoft::WRL::ComPtr<ID3D12Resource>  m_GpuTimerReadback;
        uint64_t m_GpuTimestampFreq = 0;

        float m_CpuTimeBuildSpatialIndex = 0.0f;
        float m_GpuTimeBackprop = 0.0f;
        float m_GpuTimeOptMLP = 0.0f;
        float m_GpuTimeOptFeat = 0.0f;
        float m_GpuTimeBroadcast = 0.0f;
        float m_GpuTimeRender = 0.0f;
    };
}