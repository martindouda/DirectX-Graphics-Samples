#include <imgui/imgui.h>

#include "Gate.h"
#include "Renderer.h"
#include "CompiledShaders/GateVS.h"
#include "CompiledShaders/GatePS.h"
#include "CompiledShaders/EncodeUVCS.h"
#include "CompiledShaders/GateBackpropCS.h"
#include "CompiledShaders/GateOptimizeFeaturesCS.h"
#include "CompiledShaders/GateOptimizeMLPCS.h"

using namespace Math;
using namespace Graphics;

namespace Sponza
{
    Gate::Gate() :
        m_GatePSO(L"GATE: Forward PSO"),
        m_GateBackpropPSO(L"GATE: Backprop"),
        m_GateOptMLPPSO(L"GATE: Optimize MLP"),
        m_GateOptFeatPSO(L"GATE: Optimize Features"),
        m_EncodeColorPSO(L"GATE: Encode UVs CS")
    {
    }

    Gate::~Gate()
    {
        Cleanup();
    }

    void Gate::Startup(const ModelH3D& model, DXGI_FORMAT colorFormat, DXGI_FORMAT depthFormat)
    {
		m_Model = &model;

        // -------------------------------------------------------------------------
        // 1. Setup Inference PSO
        // -------------------------------------------------------------------------
        m_GateRootSig.Reset(4, 0);
        m_GateRootSig[0].InitAsConstantBuffer(0);
        m_GateRootSig[1].InitAsBufferSRV(0);
        m_GateRootSig[2].InitAsBufferSRV(1);
        m_GateRootSig[3].InitAsConstants(1, 1);
        m_GateRootSig.Finalize(L"Gate Inference Root Sig", D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

        D3D12_INPUT_ELEMENT_DESC vertElem[] = {
            { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
            { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
            { "NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
            { "TANGENT", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
            { "BITANGENT", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 }
        };

        m_GatePSO.SetRootSignature(m_GateRootSig);
        m_GatePSO.SetRasterizerState(RasterizerDefault);
        m_GatePSO.SetBlendState(BlendDisable);
        m_GatePSO.SetDepthStencilState(DepthStateTestEqual);
        m_GatePSO.SetInputLayout(_countof(vertElem), vertElem);
        m_GatePSO.SetPrimitiveTopologyType(D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);
        m_GatePSO.SetRenderTargetFormats(1, &colorFormat, depthFormat);
        m_GatePSO.SetVertexShader(g_pGateVS, sizeof(g_pGateVS));
        m_GatePSO.SetPixelShader(g_pGatePS, sizeof(g_pGatePS));
        m_GatePSO.Finalize();

        // -------------------------------------------------------------------------
        // 2. Setup Training Root Sig & PSOs
        // -------------------------------------------------------------------------
        m_GateTrainRootSig.Reset(10, 1);
        m_GateTrainRootSig[0].InitAsConstants(0, 10);
        m_GateTrainRootSig[1].InitAsBufferSRV(0);
        m_GateTrainRootSig[2].InitAsBufferSRV(1);
        m_GateTrainRootSig[3].InitAsDescriptorTable(1);
        m_GateTrainRootSig[3].SetTableRange(0, D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, (UINT)-1, 1);
        m_GateTrainRootSig[4].InitAsBufferUAV(0);
        m_GateTrainRootSig[5].InitAsBufferUAV(1);
        m_GateTrainRootSig[6].InitAsBufferUAV(2);
        m_GateTrainRootSig[7].InitAsBufferUAV(3);
        m_GateTrainRootSig[8].InitAsBufferUAV(4);
        m_GateTrainRootSig[9].InitAsBufferUAV(5);
        m_GateTrainRootSig.InitStaticSampler(0, Graphics::SamplerLinearWrapDesc);
        m_GateTrainRootSig.Finalize(L"GATE Training Root Sig");

        m_GateBackpropPSO.SetRootSignature(m_GateTrainRootSig);
        m_GateBackpropPSO.SetComputeShader(g_pGateBackpropCS, sizeof(g_pGateBackpropCS));
        m_GateBackpropPSO.Finalize();

        m_GateOptMLPPSO.SetRootSignature(m_GateTrainRootSig);
        m_GateOptMLPPSO.SetComputeShader(g_pGateOptimizeMLPCS, sizeof(g_pGateOptimizeMLPCS));
        m_GateOptMLPPSO.Finalize();

        m_GateOptFeatPSO.SetRootSignature(m_GateTrainRootSig);
        m_GateOptFeatPSO.SetComputeShader(g_pGateOptimizeFeaturesCS, sizeof(g_pGateOptimizeFeaturesCS));
        m_GateOptFeatPSO.Finalize();

        // -------------------------------------------------------------------------
        // 3. Buffer Allocation
        // -------------------------------------------------------------------------
        uint32_t VertexStride = model.GetVertexStride();
        m_TotalVertices = model.GetVertexBuffer().SizeInBytes / VertexStride;

        // Features
        std::vector<GateFeature> initialFeatures(m_TotalVertices);
        for (uint32_t i = 0; i < m_TotalVertices; ++i)
        {
            initialFeatures[i].data[0] = DirectX::XMFLOAT4((float)rand() / RAND_MAX, (float)rand() / RAND_MAX, (float)rand() / RAND_MAX, (float)rand() / RAND_MAX);
            initialFeatures[i].data[1] = DirectX::XMFLOAT4((float)rand() / RAND_MAX, (float)rand() / RAND_MAX, (float)rand() / RAND_MAX, (float)rand() / RAND_MAX);
        }
        m_GateFeatureBuffer.Create(L"GATE Feature Buffer", m_TotalVertices, sizeof(GateFeature), initialFeatures.data());
        m_GateFeatureGradientBuffer.Create(L"GATE Feature Gradients", m_TotalVertices * 8, sizeof(float), nullptr);

        // MLP
        uint32_t numNetworkParameters = 212;
        std::vector<float> initialWeights(numNetworkParameters);
        for (uint32_t i = 0; i < numNetworkParameters; ++i)
            initialWeights[i] = ((float)rand() / (float)RAND_MAX) * 0.2f - 0.1f;

        m_GateMLPBuffer.Create(L"MLP Parameters", numNetworkParameters, sizeof(float), initialWeights.data());
        m_GateMLPGradientBuffer.Create(L"MLP Gradients", numNetworkParameters, sizeof(float), nullptr);

        // Adam
        std::vector<AdamData> initialFeatureAdam(m_TotalVertices * 2, { {0,0,0,0}, {0,0,0,0} });
        std::vector<AdamData> initialMLPAdam(53, { {0,0,0,0}, {0,0,0,0} });
        m_GateFeatureAdamBuffer.Create(L"Feature Adam Buffer", m_TotalVertices * 2, sizeof(AdamData), initialFeatureAdam.data());
        m_GateMLPAdamBuffer.Create(L"MLP Adam Buffer", 53, sizeof(AdamData), initialMLPAdam.data());

        // Global Triangles
        m_TotalTriangles = 0;
        for (uint32_t i = 0; i < model.GetMeshCount(); ++i)
            m_TotalTriangles += model.GetMesh(i).indexCount / 3;

        std::vector<GlobalTriangle> globalTris(m_TotalTriangles);
        uint32_t triOffset = 0;
        const unsigned char* rawIndexData = model.GetIndexData();

        for (uint32_t meshIndex = 0; meshIndex < model.GetMeshCount(); ++meshIndex)
        {
            const ModelH3D::Mesh& mesh = model.GetMesh(meshIndex);
            uint32_t baseVertex = mesh.vertexDataByteOffset / VertexStride;
            const uint16_t* cpuIndexData = (const uint16_t*)(rawIndexData + mesh.indexDataByteOffset);

            for (uint32_t i = 0; i < mesh.indexCount; i += 3)
            {
                globalTris[triOffset].i0 = cpuIndexData[i + 0] + baseVertex;
                globalTris[triOffset].i1 = cpuIndexData[i + 1] + baseVertex;
                globalTris[triOffset].i2 = cpuIndexData[i + 2] + baseVertex;
                globalTris[triOffset].materialIdx = mesh.materialIndex;
                triOffset++;
            }
        }
        m_GlobalTriangleBuffer.Create(L"Global Triangle Buffer", m_TotalTriangles, sizeof(GlobalTriangle), globalTris.data());
    }

    void Gate::Train(ComputeContext& trainCtx)
    {
        if (m_IsTrainingPaused)
            return;

        uint32_t uvOffset = m_Model->GetMesh(0).attrib[ModelH3D::attrib_texcoord0].offset;
        uint32_t VertexStride = m_Model->GetVertexStride();

        trainCtx.SetRootSignature(m_GateTrainRootSig);
        trainCtx.SetDescriptorHeap(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, Renderer::s_TextureHeap.GetHeapPointer());

        struct TrainingConstants {
            uint32_t trainingStep;
            uint32_t totalTriangles;
            float learningRate;
            float adamEpsilon;
            float adamBeta1;
            float adamBeta2;
            float adamBeta1T;
            float adamBeta2T;
            uint32_t VertexStride;
            uint32_t uvOffset;
        } cb = {
            m_TrainingStep, m_TotalTriangles, m_LearningRate, m_AdamEpsilon,
            m_AdamBeta1, m_AdamBeta2, m_AdamBeta1T, m_AdamBeta2T, VertexStride, uvOffset
        };
        trainCtx.SetConstantArray(0, 10, &cb);

        trainCtx.GetCommandList()->SetComputeRootShaderResourceView(1, m_GlobalTriangleBuffer.GetGpuVirtualAddress());
        trainCtx.GetCommandList()->SetComputeRootShaderResourceView(2, m_Model->GetVertexBuffer().BufferLocation);
        trainCtx.SetDescriptorTable(3, m_Model->GetSRVs(0));

        trainCtx.SetBufferUAV(4, m_GateFeatureBuffer);
        trainCtx.SetBufferUAV(5, m_GateFeatureGradientBuffer);
        trainCtx.SetBufferUAV(6, m_GateFeatureAdamBuffer);
        trainCtx.SetBufferUAV(7, m_GateMLPBuffer);
        trainCtx.SetBufferUAV(8, m_GateMLPGradientBuffer);
        trainCtx.SetBufferUAV(9, m_GateMLPAdamBuffer);

        trainCtx.SetPipelineState(m_GateBackpropPSO);
        trainCtx.Dispatch(m_BackpropDispatchedGroups, 1, 1);

        trainCtx.InsertUAVBarrier(m_GateFeatureGradientBuffer);
        trainCtx.InsertUAVBarrier(m_GateMLPGradientBuffer);

        trainCtx.SetPipelineState(m_GateOptMLPPSO);
        trainCtx.Dispatch(1, 1, 1);

        trainCtx.SetPipelineState(m_GateOptFeatPSO);
        trainCtx.Dispatch(Math::DivideByMultiple(m_TotalVertices * 2, 64), 1, 1);

        trainCtx.TransitionResource(m_GateFeatureBuffer, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        trainCtx.TransitionResource(m_GateMLPBuffer, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);

        m_TrainingStep++;
        m_AdamBeta1T *= m_AdamBeta1;
        m_AdamBeta2T *= m_AdamBeta2;
    }

    void Gate::RenderVisualization(GraphicsContext& gfxContext, const Camera& camera,
        ColorBuffer& targetBuffer, DepthBuffer& depthBuffer,
        const D3D12_VIEWPORT& viewport, const D3D12_RECT& scissor)
    {
        gfxContext.TransitionResource(targetBuffer, D3D12_RESOURCE_STATE_RENDER_TARGET, true);
        gfxContext.ClearColor(targetBuffer);

        gfxContext.SetPipelineState(m_GatePSO);
        gfxContext.SetRootSignature(m_GateRootSig);

        Matrix4 wvp = camera.GetViewProjMatrix();
        gfxContext.SetDynamicConstantBufferView(0, sizeof(wvp), &wvp);
        gfxContext.SetBufferSRV(1, m_GateFeatureBuffer);
        gfxContext.SetBufferSRV(2, m_GateMLPBuffer);

        D3D12_CPU_DESCRIPTOR_HANDLE gateRTVs[] = { targetBuffer.GetRTV() };
        gfxContext.SetRenderTargets(1, gateRTVs, depthBuffer.GetDSV_DepthReadOnly());
        gfxContext.SetViewportAndScissor(viewport, scissor);

        uint32_t VertexStride = m_Model->GetVertexStride();
        for (uint32_t meshIndex = 0; meshIndex < m_Model->GetMeshCount(); ++meshIndex)
        {
            const ModelH3D::Mesh& mesh = m_Model->GetMesh(meshIndex);
            uint32_t indexCount = mesh.indexCount;
            uint32_t startIndex = mesh.indexDataByteOffset / sizeof(uint16_t);
            uint32_t baseVertex = mesh.vertexDataByteOffset / VertexStride;

            gfxContext.SetConstants(3, baseVertex);
            gfxContext.DrawIndexed(indexCount, startIndex, baseVertex);
        }

        gfxContext.TransitionResource(targetBuffer, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, true);
    }

    void Gate::RenderGUI()
    {
        ImGui::Begin("GATE Training Configuration");

        ImGui::Text("Network Status");
        ImGui::Text("Training Step: %u", m_TrainingStep);

        ImGui::Checkbox("Pause Training", &m_IsTrainingPaused);

        if (ImGui::Button("Reset Training", ImVec2(ImGui::GetContentRegionAvail().x, 30)))
            ResetTraining();

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        ImGui::Text("Hyperparameters");

        ImGui::SliderInt("Backprop Steps", &m_BackpropDispatchedGroups, 1, 8192, "%d groups");
        // Logarithmic slider is great for learning rates
        ImGui::SliderFloat("Learning Rate", &m_LearningRate, 0.00001f, 0.05f, "%.5f", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("Adam Beta 1", &m_AdamBeta1, 0.8f, 0.999f, "%.4f");
        ImGui::SliderFloat("Adam Beta 2", &m_AdamBeta2, 0.9f, 0.9999f, "%.5f");
        ImGui::SliderFloat("Adam Epsilon", &m_AdamEpsilon, 1e-8f, 1e-4f, "%.8f", ImGuiSliderFlags_Logarithmic);

        ImGui::End();
    }

    void Gate::ResetTraining()
    {
        // 1. Reset Training State
        m_TrainingStep = 1;
        m_AdamBeta1T = m_AdamBeta1;
        m_AdamBeta2T = m_AdamBeta2;

        uint32_t VertexStride = m_Model->GetVertexStride();
        uint32_t totalVertices = m_Model->GetVertexBuffer().SizeInBytes / VertexStride;

        // 2. Re-randomize Features
        std::vector<GateFeature> initialFeatures(totalVertices);
        for (uint32_t i = 0; i < totalVertices; ++i)
        {
            initialFeatures[i].data[0] = DirectX::XMFLOAT4((float)rand() / RAND_MAX, (float)rand() / RAND_MAX, (float)rand() / RAND_MAX, (float)rand() / RAND_MAX);
            initialFeatures[i].data[1] = DirectX::XMFLOAT4((float)rand() / RAND_MAX, (float)rand() / RAND_MAX, (float)rand() / RAND_MAX, (float)rand() / RAND_MAX);
        }
        m_GateFeatureBuffer.Create(L"GATE Feature Buffer", totalVertices, sizeof(GateFeature), initialFeatures.data());

        // 3. Re-randomize MLP Parameters
        uint32_t numNetworkParameters = 212;
        std::vector<float> initialWeights(numNetworkParameters);
        for (uint32_t i = 0; i < numNetworkParameters; ++i)
        {
            initialWeights[i] = ((float)rand() / (float)RAND_MAX) * 0.2f - 0.1f;
        }
        m_GateMLPBuffer.Create(L"MLP Parameters", numNetworkParameters, sizeof(float), initialWeights.data());

        // 4. Zero out Adam and Gradient Buffers
        std::vector<AdamData> initialFeatureAdam(totalVertices * 2, { {0,0,0,0}, {0,0,0,0} });
        std::vector<AdamData> initialMLPAdam(53, { {0,0,0,0}, {0,0,0,0} });

        m_GateFeatureAdamBuffer.Create(L"Feature Adam Buffer", totalVertices * 2, sizeof(AdamData), initialFeatureAdam.data());
        m_GateMLPAdamBuffer.Create(L"MLP Adam Buffer", 53, sizeof(AdamData), initialMLPAdam.data());

        // (Gradients are zeroed on creation anyway, but we recreate to be safe)
        m_GateFeatureGradientBuffer.Create(L"GATE Feature Gradients", totalVertices * 8, sizeof(float), nullptr);
        m_GateMLPGradientBuffer.Create(L"MLP Gradients", numNetworkParameters, sizeof(float), nullptr);
    }

    void Gate::Cleanup()
    {
        m_GateFeatureBuffer.Destroy();
        m_GateFeatureGradientBuffer.Destroy();
        m_GateFeatureAdamBuffer.Destroy();
        m_GateMLPBuffer.Destroy();
        m_GateMLPGradientBuffer.Destroy();
        m_GateMLPAdamBuffer.Destroy();
        m_GlobalTriangleBuffer.Destroy();
        m_VertexMaterialMap.Destroy();
    }
}