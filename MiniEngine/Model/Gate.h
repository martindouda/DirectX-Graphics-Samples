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
            return ((k.x * 73856093) ^ (k.y * 19349663) ^ (k.z * 83492791));
        }
    };

    // Ensures we always identify an edge the same way, regardless of triangle winding order
    struct SpatialEdge
    {
        uint32_t vMin, vMax;

        SpatialEdge(uint32_t a, uint32_t b) {
            vMin = std::min(a, b);
            vMax = std::max(a, b);
        }

        bool operator==(const SpatialEdge& other) const {
            return vMin == other.vMin && vMax == other.vMax;
        }
    };

    struct SpatialEdgeHasher
    {
        std::size_t operator()(const SpatialEdge& e) const {
            return std::hash<uint32_t>()(e.vMin) ^ (std::hash<uint32_t>()(e.vMax) << 1);
        }
    };

    class Gate
    {
    public:
        Gate();
        ~Gate();

        // --- Main API ---
        void Startup(const ModelH3D& model, DXGI_FORMAT colorFormat, DXGI_FORMAT depthFormat);
        void Train(ComputeContext& trainCtx, ColorBuffer& visibilityBuffer);
        void RenderVisualization(GraphicsContext& gfxContext, const Math::Camera& camera, DepthBuffer& depthBuffer,
            const D3D12_VIEWPORT& viewport, const D3D12_RECT& scissor, ColorBuffer& visibilityBuffer);

        void RenderGUI();
        void ResetTraining();
        void Cleanup();

        inline ColorBuffer& GetGateColorBuffer() { return m_GateColorBuffer; }
        inline ColorBuffer& GetVisColorBuffer() { return m_VisColorBuffer; }
        inline void SetIsTrainingPaused(bool isTrainingPaused) { m_IsTrainingPaused = isTrainingPaused; }
        inline bool GetIsTrainingPaused() const { return m_IsTrainingPaused; }

    private:
        // --- Init helpers ---
        void BuildSpatialIndex(const ModelH3D& model);
        void AllocateBuffers(const ModelH3D& model);
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

        uint32_t m_TotalUniqueMeshColorPoints = 0;
        uint32_t m_TotalDuplicatedMeshColorPoints = 0;

        // Geometry buffers
        StructuredBuffer m_GlobalTriangleBuffer;
        StructuredBuffer m_SpatialTriangleBuffer;
        StructuredBuffer m_VertexMaterialMap;

        // Feature buffers (Cache-coherency architecture)
        StructuredBuffer m_GateFeatureBuffer;           // Duplicated   (for fast read during Inference/Backprop)
        StructuredBuffer m_UniqueFeatureBuffer;         // Unique       (for write by Adam optimizer)
        StructuredBuffer m_MeshColorMappingBuffer;      // N:M mapping  (replaces m_VertexMappingBuffer)

        ByteAddressBuffer m_GateFeatureGradientBuffer;
        ByteAddressBuffer m_GateFeatureAdamBuffer;

        // MLP buffers
        ByteAddressBuffer m_GateMLPBuffer;
        ByteAddressBuffer m_GateMLPGradientBuffer;
        ByteAddressBuffer m_GateMLPAdamBuffer;

        // --- PSOs and Root Signatures ---
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
        uint32_t m_TrainingStep = 1;
        bool m_IsTrainingPaused = true;

        int m_BackpropDispatchedGroups = 8192; // * 64 triangles per step
        float m_FeatureLearningRate = 0.05f;
        float m_MLPLearningRate = 0.002f;
        float m_AdamEpsilon = 1e-8f;
        float m_AdamBeta1 = 0.9f;
        float m_AdamBeta2 = 0.999f;
        float m_WeightDecay = 0.01f;
        float m_ScreenSpaceRatio = 0.85f;

        // Mesh Colors Configuration
        uint32_t m_MeshColorR = 2; // Resolution (1 = vertices only, 2 = 1 point per edge, etc.)
        uint32_t m_MeshColorK = 6; // Points per triangle (Calculated as: (R+1)*(R+2)/2)

        // Custom parameters for experiments
        int m_CustomInt0 = 0;
    };
}