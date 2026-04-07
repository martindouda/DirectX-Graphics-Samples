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
        uint32_t m_Resolution = 256;
        int      m_DesiredResolution = 256;        // For UI
        uint32_t m_PointsPerTri = 0;

        // Add to your Hyperparameters section:
        uint32_t m_DesiredFeatureFloats = 4; // For UI
        uint32_t m_FeatureFloats = 4;
        uint32_t m_FeatureQuartets = 1;

        // Add to track dynamic MLP size:
        uint32_t m_MlpParameterCount = 212;
        uint32_t m_MlpQuartets = 53;

        int      m_BackpropDispatchedGroups = 1024; // * 1024 triangles per step
        float    m_GlobalLearningRate = 0.1f;
        float    m_LearningRateRatio = 0.5f;
        float    m_AdamEpsilon = 1e-8f;
        float    m_AdamBeta1 = 0.9f;
        float    m_AdamBeta2 = 0.999f;
        float    m_WeightDecay = 0.01f;
        float    m_ScreenSpaceRatio = 0.85f;
        float    m_AoRadius = 150.0f;
        int      m_LightingMode = 3; // 0 = None, 1 = AO Only, 2 = Shadows Only, 3 = Both
        bool     m_TexturelessView = false;
        bool     m_DisableDirectionalLight = false;

        // --- GPU Resources: Geometry & Features ---
        StructuredBuffer  m_GlobalTriangleBuffer;
        StructuredBuffer  m_VertexMaterialMap;

        // Cache-coherency architecture buffers
        StructuredBuffer  m_GateFeatureBuffer;          // Duplicated  (for fast read during Inference/Backprop)
        StructuredBuffer  m_UniqueFeatureBuffer;        // Unique      (for write by Adam optimizer)
        StructuredBuffer  m_VertexMappingBuffer;        // N:M mapping (for copying data to duplicates)

        StructuredBuffer m_GateFeatureGradientBuffer;
        StructuredBuffer m_GateFeatureAdamBuffer;

        // --- GPU Resources: MLP ---
        StructuredBuffer m_GateMLPBuffer;
        StructuredBuffer m_GateMLPGradientBuffer;
        StructuredBuffer m_GateMLPAdamBuffer;

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