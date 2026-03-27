// Gate.cpp

#include <imgui/imgui.h>
#include <vector>
#include <unordered_map>
#include <DirectXMath.h>

#include "Gate.h"
#include "Renderer.h"
#include "CompiledShaders/GateVS.h"
#include "CompiledShaders/GatePS.h"
#include "CompiledShaders/EncodeUVCS.h"
#include "CompiledShaders/GateBackpropCS.h"
#include "CompiledShaders/GateOptimizeFeaturesCS.h"
#include "CompiledShaders/GateOptimizeMLPCS.h"
#include "CompiledShaders/VisBufferCS.h"
#include "CompiledShaders/GateBroadcastCS.h"

using namespace Math;
using namespace Graphics;

namespace Sponza
{
    Gate::Gate() :
        m_GatePSO(L"GATE: Forward PSO"), m_GateBackpropPSO(L"GATE: Backprop"), m_GateOptMLPPSO(L"GATE: Optimize MLP"),
		m_GateOptFeatPSO(L"GATE: Optimize Features"), m_EncodeColorPSO(L"GATE: Encode UVs CS"), m_Model(nullptr)
    {
    }

    Gate::~Gate()
    {
        Cleanup();
    }

    void Gate::Startup(const ModelH3D& model, DXGI_FORMAT colorFormat, DXGI_FORMAT depthFormat)
    {
        m_Model = &model;
        m_GateColorBuffer.Create(L"Gate Output Buffer", g_SceneColorBuffer.GetWidth(), g_SceneColorBuffer.GetHeight(), 1, g_SceneColorBuffer.GetFormat());

        // 1. Setup Inference PSO
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

        // 2. Setup Training Root Sig & PSOs
        m_GateTrainRootSig.Reset(14, 1);
        m_GateTrainRootSig[0].InitAsConstants(0, 14); // register(b0)
        m_GateTrainRootSig[1].InitAsBufferSRV(0);     // TriangleBuffer register(t0)
        m_GateTrainRootSig[2].InitAsBufferSRV(1);     // VertexUVBuffer register(t1)

        m_GateTrainRootSig[3].InitAsDescriptorTable(1);
        m_GateTrainRootSig[3].SetTableRange(0, D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 2, 1, 0); // VisBuffer (t2)

        m_GateTrainRootSig[4].InitAsDescriptorTable(1);
        m_GateTrainRootSig[4].SetTableRange(0, D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, (UINT)-1, 1); // Bindless

        m_GateTrainRootSig[5].InitAsBufferUAV(0); // u0
        m_GateTrainRootSig[6].InitAsBufferUAV(1); // u1
        m_GateTrainRootSig[7].InitAsBufferUAV(2); // u2
        m_GateTrainRootSig[8].InitAsBufferUAV(3); // u3
        m_GateTrainRootSig[9].InitAsBufferUAV(4); // u4
        m_GateTrainRootSig[10].InitAsBufferUAV(5); // u5

        // NOVÉ SLOTY PRO BROADCAST SHADER:
        m_GateTrainRootSig[11].InitAsBufferSRV(3); // t3: VertexMappingBuffer
        m_GateTrainRootSig[12].InitAsBufferSRV(4); // t4: UniqueFeatureBuffer
        m_GateTrainRootSig[13].InitAsBufferSRV(5); // t5: SpatialTriangleBuffer

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

        // INICIALIZACE BROADCAST PSO:
        m_GateBroadcastPSO.SetRootSignature(m_GateTrainRootSig);
        m_GateBroadcastPSO.SetComputeShader(g_pGateBroadcastCS, sizeof(g_pGateBroadcastCS));
        m_GateBroadcastPSO.Finalize();

        // 1. Create the UI-friendly texture (UNORM format)
        m_VisColorBuffer.Create(L"Visibility Vis Buffer", g_SceneColorBuffer.GetWidth(), g_SceneColorBuffer.GetHeight(), 1, DXGI_FORMAT_R8G8B8A8_UNORM);

        // 2. Setup the Root Signature
        m_VisRootSig.Reset(2, 0);
        m_VisRootSig[0].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 1); // t0
        m_VisRootSig[1].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 0, 1); // u0
        m_VisRootSig.Finalize(L"Vis Buffer Root Sig");

        // 3. Setup the PSO
        m_VisPSO.SetRootSignature(m_VisRootSig);
        m_VisPSO.SetComputeShader(g_pVisBufferCS, sizeof(g_pVisBufferCS));
        m_VisPSO.Finalize();

        uint32_t vertexStride = model.GetVertexStride();
        m_TotalVertices = model.GetVertexBuffer().SizeInBytes / vertexStride;

        // Kvantizace (Welding)
        const float QUANTIZATION_FACTOR = 10000.0f;
        std::unordered_map<Int3, uint32_t, Int3Hash> spatialHashMap;
        std::vector<uint32_t> originalToSpatialMap(m_TotalVertices);
        m_UniqueSpatialVertexCount = 0; // POZOR: Zmìnìno na member promìnnou!

        const unsigned char* rawVertexData = m_Model->GetVertexData();
        for (uint32_t i = 0; i < m_TotalVertices; ++i)
        {
            DirectX::XMFLOAT3* pos = (DirectX::XMFLOAT3*)(rawVertexData + (i * vertexStride));

            Int3 qPos;
            qPos.x = static_cast<int32_t>(std::round(pos->x * QUANTIZATION_FACTOR));
            qPos.y = static_cast<int32_t>(std::round(pos->y * QUANTIZATION_FACTOR));
            qPos.z = static_cast<int32_t>(std::round(pos->z * QUANTIZATION_FACTOR));

            auto it = spatialHashMap.find(qPos);
            if (it != spatialHashMap.end())
            {
                originalToSpatialMap[i] = it->second;
            }
            else
            {
                spatialHashMap[qPos] = m_UniqueSpatialVertexCount;
                originalToSpatialMap[i] = m_UniqueSpatialVertexCount;
                m_UniqueSpatialVertexCount++;
            }
        }

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
            uint32_t baseVertex = mesh.vertexDataByteOffset / vertexStride;
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

        // Spatial Index Buffer
        std::vector<GlobalTriangle> spatialGlobalTris(m_TotalTriangles);
        for (uint32_t i = 0; i < m_TotalTriangles; ++i)
        {
            GlobalTriangle oldTri = globalTris[i];
            GlobalTriangle newTri;
            newTri.i0 = originalToSpatialMap[oldTri.i0];
            newTri.i1 = originalToSpatialMap[oldTri.i1];
            newTri.i2 = originalToSpatialMap[oldTri.i2];
            newTri.materialIdx = oldTri.materialIdx;
            spatialGlobalTris[i] = newTri;
        }
        m_SpatialTriangleBuffer.Create(L"Spatial Triangle Buffer", m_TotalTriangles, sizeof(GlobalTriangle), spatialGlobalTris.data());

        // Vytvoøení VertexMappingBufferu
        m_VertexMappingBuffer.Create(L"Vertex Mapping Buffer", m_TotalVertices, sizeof(uint32_t), originalToSpatialMap.data());


        // A. Duplikovaný Feature Buffer
        std::vector<GateFeature> duplicatedFeatures(m_TotalVertices);
        for (uint32_t i = 0; i < m_TotalVertices; ++i)
        {
            duplicatedFeatures[i].data[0] = DirectX::XMFLOAT4((float)rand() / RAND_MAX, (float)rand() / RAND_MAX, (float)rand() / RAND_MAX, (float)rand() / RAND_MAX);
            duplicatedFeatures[i].data[1] = DirectX::XMFLOAT4((float)rand() / RAND_MAX, (float)rand() / RAND_MAX, (float)rand() / RAND_MAX, (float)rand() / RAND_MAX);
        }
        m_GateFeatureBuffer.Create(L"DUPLICATED Feature Buffer", m_TotalVertices, sizeof(GateFeature), duplicatedFeatures.data());

        // B. Unikátní Feature Buffer a tréninkové buffery
        // Initial features pro unikátní (mùže být stejné random)
        std::vector<GateFeature> uniqueFeatures(m_UniqueSpatialVertexCount);
        for (uint32_t i = 0; i < m_UniqueSpatialVertexCount; ++i)
        {
            uniqueFeatures[i].data[0] = DirectX::XMFLOAT4((float)rand() / RAND_MAX, (float)rand() / RAND_MAX, (float)rand() / RAND_MAX, (float)rand() / RAND_MAX);
            uniqueFeatures[i].data[1] = DirectX::XMFLOAT4((float)rand() / RAND_MAX, (float)rand() / RAND_MAX, (float)rand() / RAND_MAX, (float)rand() / RAND_MAX);
        }
        m_UniqueFeatureBuffer.Create(L"UNIQUE Feature Buffer", m_UniqueSpatialVertexCount, sizeof(GateFeature), uniqueFeatures.data());

        std::vector<AdamData> initialFeatureAdam(m_UniqueSpatialVertexCount * 2, { {0,0,0,0}, {0,0,0,0}, 0, {0,0,0} });
        m_GateFeatureAdamBuffer.Create(L"UNIQUE Feature Adam Buffer", m_UniqueSpatialVertexCount * 2, sizeof(AdamData), initialFeatureAdam.data());
        m_GateFeatureGradientBuffer.Create(L"UNIQUE Feature Gradients", m_UniqueSpatialVertexCount * 8, sizeof(float), nullptr);

        // MLP
        uint32_t numNetworkParameters = 212;
        std::vector<float> initialWeights(numNetworkParameters);
        for (uint32_t i = 0; i < numNetworkParameters; ++i)
            initialWeights[i] = ((float)rand() / (float)RAND_MAX) * 0.2f - 0.1f;

        m_GateMLPBuffer.Create(L"MLP Parameters", numNetworkParameters, sizeof(float), initialWeights.data());
        m_GateMLPGradientBuffer.Create(L"MLP Gradients", numNetworkParameters, sizeof(float), nullptr);

        std::vector<AdamData> initialMLPAdam(53, { {0,0,0,0}, {0,0,0,0}, 0, {0,0,0} });
        m_GateMLPAdamBuffer.Create(L"MLP Adam Buffer", 53, sizeof(AdamData), initialMLPAdam.data());
    }

    void Gate::Train(ComputeContext& trainCtx, ColorBuffer& visibilityBuffer)
    {
        if (m_IsTrainingPaused)
            return;

        uint32_t uvOffset = m_Model->GetMesh(0).attrib[ModelH3D::attrib_texcoord0].offset;
        uint32_t VertexStride = m_Model->GetVertexStride();

        trainCtx.SetRootSignature(m_GateTrainRootSig);
        trainCtx.SetDescriptorHeap(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, Renderer::s_TextureHeap.GetHeapPointer());

        struct TrainingConstants
        {
            uint32_t trainingStep;
            uint32_t totalTriangles;
            float featureLearningRate;
            float mlpLearningRate;
            float adamEpsilon;
            float adamBeta1;
            float adamBeta2;
            float weightDecay;
            float screenSpaceRatio;
            uint32_t VertexStride;
            uint32_t uvOffset;
            uint32_t screenWidth;
            uint32_t screenHeight;
            int CustomInt0;
        } cb = {
            m_TrainingStep, m_TotalTriangles, m_FeatureLearningRate, m_MLPLearningRate, m_AdamEpsilon,
            m_AdamBeta1, m_AdamBeta2, m_WeightDecay, m_ScreenSpaceRatio, VertexStride, uvOffset,
            (uint32_t)g_SceneColorBuffer.GetWidth(), (uint32_t)g_SceneColorBuffer.GetHeight(), m_CustomInt0
        };
        trainCtx.SetConstantArray(0, 14, &cb);

        // --- BACKPROP SETUP ---
        // Pùvodní geometrie (t0 = slot 1) a Vertex data (t1 = slot 2) pro ètení UVs
        trainCtx.GetCommandList()->SetComputeRootShaderResourceView(1, m_GlobalTriangleBuffer.GetGpuVirtualAddress());
        trainCtx.GetCommandList()->SetComputeRootShaderResourceView(2, m_Model->GetVertexBuffer().BufferLocation);

        // Prostorová geometrie (t5 = slot 13) pro sí
        trainCtx.GetCommandList()->SetComputeRootShaderResourceView(13, m_SpatialTriangleBuffer.GetGpuVirtualAddress());

        trainCtx.SetDynamicDescriptor(3, 0, visibilityBuffer.GetSRV());
        trainCtx.SetDescriptorTable(4, m_Model->GetSRVs(0));

        // Zde dáváme m_UniqueFeatureBuffer na u0 (slot 5) pro optimalizaci a zápis gradientù
        trainCtx.SetBufferUAV(5, m_UniqueFeatureBuffer);
        trainCtx.SetBufferUAV(6, m_GateFeatureGradientBuffer);
        trainCtx.SetBufferUAV(7, m_GateFeatureAdamBuffer);

        trainCtx.SetBufferUAV(8, m_GateMLPBuffer);
        trainCtx.SetBufferUAV(9, m_GateMLPGradientBuffer);
        trainCtx.SetBufferUAV(10, m_GateMLPAdamBuffer);

        // 1. Backprop
        trainCtx.SetPipelineState(m_GateBackpropPSO);
        trainCtx.Dispatch(m_BackpropDispatchedGroups, 1, 1);

        trainCtx.InsertUAVBarrier(m_GateFeatureGradientBuffer);
        trainCtx.InsertUAVBarrier(m_GateMLPGradientBuffer);

        // 2. Optimize MLP
        trainCtx.SetPipelineState(m_GateOptMLPPSO);
        trainCtx.Dispatch(1, 1, 1);

        // 3. Optimize Features (Bìží nad Unique Vertex Count!)
        trainCtx.SetPipelineState(m_GateOptFeatPSO);
        trainCtx.Dispatch(Math::DivideByMultiple(m_UniqueSpatialVertexCount * 2, 64), 1, 1);

        // Èekáme, až Adam dopíše do UniqueFeatureBufferu
        trainCtx.InsertUAVBarrier(m_UniqueFeatureBuffer);

        // --- 4. BROADCAST KROK ---
        trainCtx.SetPipelineState(m_GateBroadcastPSO);

        // ZDE BYLA CHYBA: Musíš explicitnì nabindovat SRVs pro Broadcast shader!
        // t3 = slot 11 (Mapping buffer)
        trainCtx.GetCommandList()->SetComputeRootShaderResourceView(11, m_VertexMappingBuffer.GetGpuVirtualAddress());
        // t4 = slot 12 (Natrénované unikátní vlastnosti)
        trainCtx.GetCommandList()->SetComputeRootShaderResourceView(12, m_UniqueFeatureBuffer.GetGpuVirtualAddress());

        // UAV: Broadcast zapisuje do DuplicatedFeatureBuffer (u0 = slot 5)
        trainCtx.SetBufferUAV(5, m_GateFeatureBuffer);

        trainCtx.Dispatch(Math::DivideByMultiple(m_TotalVertices, 64), 1, 1);

        // Pøechody pro renderování
        trainCtx.TransitionResource(m_GateFeatureBuffer, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        trainCtx.TransitionResource(m_GateMLPBuffer, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);

        m_TrainingStep++;
    }

    void Gate::RenderVisualization(GraphicsContext& gfxContext, const Camera& camera, DepthBuffer& depthBuffer,
        const D3D12_VIEWPORT& viewport, const D3D12_RECT& scissor, ColorBuffer& visibilityBuffer)
    {
        ComputeContext& cptCtx = gfxContext.GetComputeContext();
        cptCtx.SetRootSignature(m_VisRootSig);
        cptCtx.SetPipelineState(m_VisPSO);

        cptCtx.TransitionResource(visibilityBuffer, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        cptCtx.TransitionResource(m_VisColorBuffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

        cptCtx.SetDynamicDescriptor(0, 0, visibilityBuffer.GetSRV());
        cptCtx.SetDynamicDescriptor(1, 0, m_VisColorBuffer.GetUAV());

        uint32_t dispatchX = Math::DivideByMultiple(visibilityBuffer.GetWidth(), 8);
        uint32_t dispatchY = Math::DivideByMultiple(visibilityBuffer.GetHeight(), 8);
        cptCtx.Dispatch(dispatchX, dispatchY, 1);

        cptCtx.TransitionResource(m_VisColorBuffer, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);



        gfxContext.TransitionResource(m_GateColorBuffer, D3D12_RESOURCE_STATE_RENDER_TARGET, true);
        gfxContext.ClearColor(m_GateColorBuffer);

        gfxContext.SetPipelineState(m_GatePSO);
        gfxContext.SetRootSignature(m_GateRootSig);

        Matrix4 wvp = camera.GetViewProjMatrix();
        gfxContext.SetDynamicConstantBufferView(0, sizeof(wvp), &wvp);
        gfxContext.SetBufferSRV(1, m_GateFeatureBuffer);
        gfxContext.SetBufferSRV(2, m_GateMLPBuffer);

        D3D12_CPU_DESCRIPTOR_HANDLE gateRTVs[] = { m_GateColorBuffer.GetRTV() };
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

        gfxContext.TransitionResource(m_GateColorBuffer, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, true);
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
        ImGui::SliderFloat("Feature Learning Rate", &m_FeatureLearningRate, 0.0001f, 0.1f, "%.5f");
        ImGui::SliderFloat("MLP Learning Rate", &m_MLPLearningRate, 0.00001f, 0.01f, "%.6f");
        ImGui::SliderFloat("Screen Space Ratio", &m_ScreenSpaceRatio, 0.0f, 1.0f, "%.2f");
        //ImGui::SliderFloat("Adam Beta 1", &m_AdamBeta1, 0.8f, 0.999f, "%.4f");
        //ImGui::SliderFloat("Adam Beta 2", &m_AdamBeta2, 0.9f, 0.9999f, "%.5f");
        //ImGui::SliderFloat("Adam Epsilon", &m_AdamEpsilon, 1e-8f, 1e-4f, "%.8f", ImGuiSliderFlags_Logarithmic);

		ImGui::SliderInt("Custom Int 0", &m_CustomInt0, 0, 100000);
        ImGui::End();
    }

    void Gate::ResetTraining()
    {
        // 1. Reset Training State
        m_TrainingStep = 1;

        uint32_t vertexStride = m_Model->GetVertexStride();
        uint32_t totalVertices = m_Model->GetVertexBuffer().SizeInBytes / vertexStride;

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
        // Initialize mean, variance, stepCount, and padding to 0
        std::vector<AdamData> initialFeatureAdam(m_TotalVertices * 2, { {0,0,0,0}, {0,0,0,0}, 0, {0,0,0} });
        std::vector<AdamData> initialMLPAdam(53, { {0,0,0,0}, {0,0,0,0}, 0, {0,0,0} });

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