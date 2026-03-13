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
    class Gate
    {
    public:
        Gate();
        ~Gate();

        // Sets up all buffers, PSOs, and initial random weights
        void Startup(const ModelH3D& model, DXGI_FORMAT colorFormat, DXGI_FORMAT depthFormat);

        // Dispatches the compute shaders to backprop and optimize
        void Train(ComputeContext& trainCtx);

        // Renders the forward pass (inference) to a target buffer
        void RenderVisualization(GraphicsContext& gfxContext, const Math::Camera& camera,
            ColorBuffer& targetBuffer, DepthBuffer& depthBuffer,
            const D3D12_VIEWPORT& viewport, const D3D12_RECT& scissor);

        // Draws the ImGui interface
        void RenderGUI();

        // Resets the neural network and Adam states back to step 1
        void ResetTraining();


        void Cleanup();

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
        };

		const ModelH3D* m_Model;

        // --- Network State & Buffers ---
        uint32_t m_TotalVertices = 0;
        uint32_t m_TotalTriangles = 0;

        StructuredBuffer m_GateFeatureBuffer;
        ByteAddressBuffer m_GateFeatureGradientBuffer;
        ByteAddressBuffer m_GateFeatureAdamBuffer;

        ByteAddressBuffer m_GateMLPBuffer;
        ByteAddressBuffer m_GateMLPGradientBuffer;
        ByteAddressBuffer m_GateMLPAdamBuffer;

        StructuredBuffer m_GlobalTriangleBuffer;
        StructuredBuffer m_VertexMaterialMap;

        // --- PSOs and Root Signatures ---
        // Inference
        GraphicsPSO m_GatePSO;
        RootSignature m_GateRootSig;

        // Training
        RootSignature m_GateTrainRootSig;
        ComputePSO m_GateBackpropPSO;
        ComputePSO m_GateOptMLPPSO;
        ComputePSO m_GateOptFeatPSO;

        // Utils
        RootSignature m_EncodeColorRootSig;
        ComputePSO m_EncodeColorPSO;

        // --- Hyperparameters & Training State ---
        uint32_t m_TrainingStep = 1;
        float m_AdamBeta1T = 0.9f;
        float m_AdamBeta2T = 0.999f;

        float m_LearningRate = 0.001f;
        float m_AdamEpsilon = 1e-8f;
        float m_AdamBeta1 = 0.9f;
        float m_AdamBeta2 = 0.999f;

        bool m_IsTrainingPaused = false;
		int m_BackpropDispatchedGroups = 1024; // * 64 triangles per step
    };
}