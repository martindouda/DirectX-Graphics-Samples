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

    class Gate
    {
    public:
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

    public:
        Gate();
        ~Gate();

        // --- Lifecycle ---
		void LoadModel(ModelH3D& model);
        void Startup(DXGI_FORMAT colorFormat, DXGI_FORMAT depthFormat);
        void Cleanup();

        // --- Execution ---
        void Train(ComputeContext& trainCtx, ColorBuffer& visibilityBuffer, Math::Vector3 sunDirection);
        void RenderInference(GraphicsContext& gfxContext, const Math::Camera& camera, DepthBuffer& depthBuffer,
            const D3D12_VIEWPORT& viewport, const D3D12_RECT& scissor, ColorBuffer& visibilityBuffer,
            Math::Vector3 sunDirection, float sunIntensity);
        void RenderGUI();
        void ResetTraining();

        // --- Accessors ---
        inline ColorBuffer& GetGateColorBuffer() { return m_GateColorBuffer; }

        inline void SetIsTrainingPaused(bool paused) { m_Config.isTrainingPaused = paused; }
        inline bool GetIsTrainingPaused() const { return m_Config.isTrainingPaused; }
        inline void SetTexturesEnabled(bool enabled) { m_Config.texturesEnabled = enabled; }
        inline bool GetTexturesEnabled() const { return m_Config.texturesEnabled; }
        inline void SetDirectionalLightEnabled(bool enabled) { m_Config.directionalLightEnabled = enabled; }
        inline bool GetDirectionalLightEnabled() const { return m_Config.directionalLightEnabled; }
        inline void SetShowSubdivisionGrid(bool show) { m_Config.showSubdivisionGrid = show; }
        inline bool GetShowSubdivisionGrid() const { return m_Config.showSubdivisionGrid; }
        inline void SetShowGroundTruthEnabled(bool show) { m_Config.showGroundTruth = show; }
        inline bool GetShowGroundTruthEnabled() const { return m_Config.showGroundTruth; }

    private:
        // --- Initialization Helpers ---
        void BuildSpatialIndex();
        void CalculateMeshEdgeLengths(std::vector<float>& outMeshMaxEdges, std::vector<float>& outMeshAvgEdges, float& outGlobalMaxEdge, float& outGlobalMaxAvgEdge);
        void GenerateQuantizedPoints(const std::vector<float>& meshMaxEdges, const std::vector<float>& meshAvgEdges, float globalMaxEdge,
            float globalMaxAvgEdge, std::vector<GlobalTriangle>& outGlobalTris, std::vector<Int3>& outQuantizedPositions, std::vector<uint32_t>& outPointToMeshMap);
        void BuildInvertedSpatialIndex(std::vector<Int3>& quantizedPositions, std::vector<uint32_t>& pointToMeshMap, std::vector<uint32_t>& outDuplicateToUniqueMap);
        void AllocateBuffers();
        void InitializePSOs(DXGI_FORMAT colorFormat, DXGI_FORMAT depthFormat);

        // --- Profiling & Stats Helpers ---
        void ReadbackGpuTimers();
        void UpdateLossHistory(float frameAverageLoss);

        // --- Core State ---
        const ModelH3D* m_Model = nullptr;

        // --- Geometry ---
        uint32_t m_TotalVertices = 0;
        uint32_t m_TotalTriangles = 0;
        uint32_t m_TotalMeshColorPoints = 0;
        uint32_t m_UniqueSpatialVertexCount = 0;

        struct GateConfig
        {
            bool     isTrainingPaused = true;
            uint32_t resolution = 32;
            int      desiredResolution = 0;
            bool     useMaxEdgeLength = false;
            bool     desiredUseMaxEdgeLength = false;
            int      learningMode = 0;

            uint32_t featureFloats = 4;
            uint32_t desiredFeatureFloats = 0;

            bool     useDeduplication = true;
            bool     desiredUseDeduplication = true;

            int      backpropDispatchedGroups = 1024;
            float    globalLearningRate = 0.05f;
            float    learningRateRatio = 0.5f;
            float    maxGradientClip = 1.0f;
            float    adamEpsilon = 1e-8f;
            float    adamBeta1 = 0.9f;
            float    adamBeta2 = 0.999f;
            float    weightDecay = 0.01f;
            float    screenSpaceRatio = 0.85f;
            float    aoRadius = 150.f;
            int      aoSamples = 1;
            int      shadowSamples = 1;
            float    shadowSoftnessAngle = 0.0f;
            int      lightingMode = 3;
            bool     texturesEnabled = true;
            bool     directionalLightEnabled = true;
            bool     showSubdivisionGrid = false;
            bool     showGroundTruth = false;

            bool enableAutoPauseTarget = false;
            bool enableAutoPauseSteps = false;
            float autoPauseThreshold = 0.02f;
        };

        GateConfig m_Config;
        uint32_t m_TrainingStep = 1;
        uint32_t m_FeatureQuartets = 1;
        uint32_t m_MlpParameterCount = 212;
        uint32_t m_MlpQuartets = 53;

        // --- GPU Resources: Geometry & Features ---
        StructuredBuffer m_GlobalTriangleBuffer;            // Triangle metadata (indices, resolution, point offsets)
        StructuredBuffer m_DuplicatedFeatureBuffer;         // Duplicated features for fast O(1) read during Inference
        StructuredBuffer m_UniqueFeatureBuffer;             // Unique features for safe atomic writes during Backprop
        StructuredBuffer m_VertexMappingBuffer;             // Maps linear point indices to unique feature IDs

        StructuredBuffer m_UniqueFeatureGradientBuffer;     // Accumulated loss gradients for unique features
        StructuredBuffer m_UniqueFeatureAdamBuffer;         // AdamW optimizer state (mean, variance) for features

        // --- GPU Resources: MLP ---
        StructuredBuffer m_MLPBuffer;                       // Trainable weights and biases of the MLP network
        StructuredBuffer m_MLPGradientBuffer;               // Accumulated loss gradients for MLP weights
        StructuredBuffer m_MLPAdamBuffer;                   // AdamW optimizer state for MLP weights

        // --- GPU Resources: Sparse "Push" Broadcast Architecture ---
        StructuredBuffer m_UniqueFeatureDirtyBuffer;        // Flags indicating which unique features were updated this frame
        StructuredBuffer m_UniqueToDuplicateOffsetBuffer;   // Inverted index: start offset for duplicates
        StructuredBuffer m_UniqueToDuplicateCountBuffer;    // Inverted index: total duplicates per unique feature
        StructuredBuffer m_DuplicateIndicesBuffer;          // Inverted index: flat array of all duplicate linear indices

        // --- Root Signatures & Pipeline States ---
        RootSignature     m_GateInferenceRootSig;
        GraphicsPSO       m_GatePSO;
        ColorBuffer       m_GateColorBuffer;

        RootSignature     m_GateTrainRootSig;
        ComputePSO        m_GateBackpropPSO;
        ComputePSO        m_GateOptMLPPSO;
        ComputePSO        m_GateOptFeatPSO;
        ComputePSO        m_GateBroadcastPSO;

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