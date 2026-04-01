// Gate.cpp

#include <imgui/imgui.h>
#include <vector>
#include <unordered_map>
#include <DirectXMath.h>
#include <chrono>
#include <algorithm>

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
        m_GateOptFeatPSO(L"GATE: Optimize Features"), m_EncodeColorPSO(L"GATE: Encode UVs CS"), m_Model(nullptr), m_PointsPerTri(0)
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
        m_VisColorBuffer.Create(L"Visibility Vis Buffer", g_SceneColorBuffer.GetWidth(), g_SceneColorBuffer.GetHeight(), 1, DXGI_FORMAT_R8G8B8A8_UNORM);

        m_LossHistory.resize(MAX_LOSS_HISTORY, 0.0f);
        m_LossBuffer.Create(L"Loss Buffer", 1, 4);
        m_LossReadbackBuffer.Create(L"Loss Readback", 1, 4);

        BuildSpatialIndex();
        AllocateBuffers();
        InitializePSOs(colorFormat, depthFormat);
    }

    void Gate::BuildSpatialIndex()
    {
        uint32_t vertexStride = m_Model->GetVertexStride();
        m_TotalVertices = m_Model->GetVertexBuffer().SizeInBytes / vertexStride;

        m_TotalTriangles = 0;
        for (uint32_t i = 0; i < m_Model->GetMeshCount(); ++i)
            m_TotalTriangles += m_Model->GetMesh(i).indexCount / 3;

        // 1. Spoèítáme body na trojúhelník podle aktuálního m_Resolution
        m_PointsPerTri = (m_Resolution + 1) * (m_Resolution + 2) / 2;
        uint32_t totalMeshColorPoints = m_TotalTriangles * m_PointsPerTri;

        std::unordered_map<Int3, uint32_t, Int3Hash> spatialHashMap;
        std::vector<uint32_t> duplicateToUniqueMap(totalMeshColorPoints);
        std::vector<GlobalTriangle> globalTris(m_TotalTriangles);

        m_UniqueSpatialVertexCount = 0;

        const float QUANTIZATION_FACTOR = 10000.0f;
        const unsigned char* rawVertexData = m_Model->GetVertexData();
        const unsigned char* rawIndexData = m_Model->GetIndexData();

        // 2. DYNAMICKÉ GENEROVÁNÍ BARYCENTRIK PRO JAKÉKOLIV R
        std::vector<DirectX::XMFLOAT3> bary(m_PointsPerTri);
        uint32_t idx = 0;
        for (uint32_t i = 0; i <= m_Resolution; ++i) {
            for (uint32_t j = 0; j <= m_Resolution - i; ++j) {
                uint32_t k = m_Resolution - i - j;
                bary[idx].x = (float)i / m_Resolution;
                bary[idx].y = (float)j / m_Resolution;
                bary[idx].z = (float)k / m_Resolution;
                idx++;
            }
        }

        uint32_t triOffset = 0;

        for (uint32_t meshIndex = 0; meshIndex < m_Model->GetMeshCount(); ++meshIndex)
        {
            const ModelH3D::Mesh& mesh = m_Model->GetMesh(meshIndex);
            uint32_t baseVertex = mesh.vertexDataByteOffset / vertexStride;
            const uint16_t* cpuIndexData = (const uint16_t*)(rawIndexData + mesh.indexDataByteOffset);

            for (uint32_t i = 0; i < mesh.indexCount; i += 3)
            {
                uint32_t i0 = cpuIndexData[i + 0] + baseVertex;
                uint32_t i1 = cpuIndexData[i + 1] + baseVertex;
                uint32_t i2 = cpuIndexData[i + 2] + baseVertex;

                // PØIDÁNO: Uložení dat pro UV souøadnice a textury
                globalTris[triOffset].i0 = i0;
                globalTris[triOffset].i1 = i1;
                globalTris[triOffset].i2 = i2;
                globalTris[triOffset].materialIdx = mesh.materialIndex;

                DirectX::XMFLOAT3* p0 = (DirectX::XMFLOAT3*)(rawVertexData + (i0 * vertexStride));
                DirectX::XMFLOAT3* p1 = (DirectX::XMFLOAT3*)(rawVertexData + (i1 * vertexStride));
                DirectX::XMFLOAT3* p2 = (DirectX::XMFLOAT3*)(rawVertexData + (i2 * vertexStride));

                for (uint32_t pt = 0; pt < m_PointsPerTri; ++pt)
                {
                    DirectX::XMFLOAT3 pos;
                    pos.x = bary[pt].x * p0->x + bary[pt].y * p1->x + bary[pt].z * p2->x;
                    pos.y = bary[pt].x * p0->y + bary[pt].y * p1->y + bary[pt].z * p2->y;
                    pos.z = bary[pt].x * p0->z + bary[pt].y * p1->z + bary[pt].z * p2->z;

                    Int3 qPos;
                    qPos.x = static_cast<int32_t>(std::round(pos.x * QUANTIZATION_FACTOR));
                    qPos.y = static_cast<int32_t>(std::round(pos.y * QUANTIZATION_FACTOR));
                    qPos.z = static_cast<int32_t>(std::round(pos.z * QUANTIZATION_FACTOR));

                    auto it = spatialHashMap.find(qPos);
                    uint32_t uniqueID;
                    if (it != spatialHashMap.end()) {
                        uniqueID = it->second;
                    }
                    else {
                        uniqueID = m_UniqueSpatialVertexCount;
                        spatialHashMap[qPos] = uniqueID;
                        m_UniqueSpatialVertexCount++;
                    }

                    uint32_t globalPointIndex = triOffset * m_PointsPerTri + pt;
                    duplicateToUniqueMap[globalPointIndex] = uniqueID;
                }
                triOffset++;
            }
        }

        m_VertexMappingBuffer.Create(L"Mesh Colors Mapping Buffer", totalMeshColorPoints, sizeof(uint32_t), duplicateToUniqueMap.data());
        
        // PØIDÁNO: Vytvoøení bufferu, aby ho GPU mohlo èíst
        m_GlobalTriangleBuffer.Create(L"Global Triangle Buffer", m_TotalTriangles, sizeof(GlobalTriangle), globalTris.data());
    }

    void Gate::AllocateBuffers()
    {
        // Celkový poèet "rozbalených" bodù (6 na každý trojúhelník)
        uint32_t totalMeshColorPoints = m_TotalTriangles * m_PointsPerTri;

        // A. DUPLIKOVANÝ BUFFER (Inference / Ètení v Pixel Shaderu)
        std::vector<GateFeature> duplicatedFeatures(totalMeshColorPoints);
        for (uint32_t i = 0; i < totalMeshColorPoints; ++i)
        {
            duplicatedFeatures[i].data[0] = DirectX::XMFLOAT4((float)rand() / RAND_MAX, (float)rand() / RAND_MAX, (float)rand() / RAND_MAX, (float)rand() / RAND_MAX);
            duplicatedFeatures[i].data[1] = DirectX::XMFLOAT4((float)rand() / RAND_MAX, (float)rand() / RAND_MAX, (float)rand() / RAND_MAX, (float)rand() / RAND_MAX);
        }
        m_GateFeatureBuffer.Create(L"DUPLICATED Feature Buffer", totalMeshColorPoints, sizeof(GateFeature), duplicatedFeatures.data());

        // B. UNIKÁTNÍ BUFFERY (Trénink / Backprop / Optimalizace)
        std::vector<GateFeature> uniqueFeatures(m_UniqueSpatialVertexCount);
        for (uint32_t i = 0; i < m_UniqueSpatialVertexCount; ++i)
        {
            uniqueFeatures[i].data[0] = DirectX::XMFLOAT4((float)rand() / RAND_MAX, (float)rand() / RAND_MAX, (float)rand() / RAND_MAX, (float)rand() / RAND_MAX);
            uniqueFeatures[i].data[1] = DirectX::XMFLOAT4((float)rand() / RAND_MAX, (float)rand() / RAND_MAX, (float)rand() / RAND_MAX, (float)rand() / RAND_MAX);
        }
        m_UniqueFeatureBuffer.Create(L"UNIQUE Feature Buffer", m_UniqueSpatialVertexCount, sizeof(GateFeature), uniqueFeatures.data());

        // Adam optimalizátor a gradienty pracují POUZE s unikátními daty
        std::vector<AdamData> initialFeatureAdam(m_UniqueSpatialVertexCount * 2, { {0,0,0,0}, {0,0,0,0}, 0, {0,0,0} });
        m_GateFeatureAdamBuffer.Create(L"UNIQUE Feature Adam Buffer", m_UniqueSpatialVertexCount * 2, sizeof(AdamData), initialFeatureAdam.data());
        m_GateFeatureGradientBuffer.Create(L"UNIQUE Feature Gradients", m_UniqueSpatialVertexCount * 8, sizeof(float), nullptr);

        // C. MLP PARAMETRY (Zùstávají beze zmìny)
        uint32_t numNetworkParameters = 212;
        std::vector<float> initialWeights(numNetworkParameters);
        for (uint32_t i = 0; i < numNetworkParameters; ++i)
            initialWeights[i] = ((float)rand() / (float)RAND_MAX) * 0.2f - 0.1f;

        m_GateMLPBuffer.Create(L"MLP Parameters", numNetworkParameters, sizeof(float), initialWeights.data());
        m_GateMLPGradientBuffer.Create(L"MLP Gradients", numNetworkParameters, sizeof(float), nullptr);

        std::vector<AdamData> initialMLPAdam(53, { {0,0,0,0}, {0,0,0,0}, 0, {0,0,0} });
        m_GateMLPAdamBuffer.Create(L"MLP Adam Buffer", 53, sizeof(AdamData), initialMLPAdam.data());
    }

    void Gate::InitializePSOs(DXGI_FORMAT colorFormat, DXGI_FORMAT depthFormat)
    {
        // 1. Setup Inference PSO
        m_GateRootSig.Reset(5, 0);
        m_GateRootSig[0].InitAsConstantBuffer(0);
        m_GateRootSig[1].InitAsBufferSRV(0);
        m_GateRootSig[2].InitAsBufferSRV(1);
        m_GateRootSig[3].InitAsConstants(1, 3);
        m_GateRootSig[4].InitAsBufferSRV(2);
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
        m_GateTrainRootSig[0].InitAsConstants(0, 15); // register(b0)
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
        m_GateTrainRootSig[11].InitAsBufferUAV(6); // u6

        m_GateTrainRootSig[12].InitAsBufferSRV(3); // t3: VertexMappingBuffer
        m_GateTrainRootSig[13].InitAsBufferSRV(4); // t4: UniqueFeatureBuffer


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

        m_GateBroadcastPSO.SetRootSignature(m_GateTrainRootSig);
        m_GateBroadcastPSO.SetComputeShader(g_pGateBroadcastCS, sizeof(g_pGateBroadcastCS));
        m_GateBroadcastPSO.Finalize();

        // 3. Setup Vis Buffer Root Sig
        m_VisRootSig.Reset(2, 0);
        m_VisRootSig[0].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 1); // t0
        m_VisRootSig[1].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 0, 1); // u0
        m_VisRootSig.Finalize(L"Vis Buffer Root Sig");

        m_VisPSO.SetRootSignature(m_VisRootSig);
        m_VisPSO.SetComputeShader(g_pVisBufferCS, sizeof(g_pVisBufferCS));
        m_VisPSO.Finalize();
    }

    void Gate::Train(ComputeContext& trainCtx, ColorBuffer& visibilityBuffer)
    {
        if (m_IsTrainingPaused)
            return;

        uint32_t zero = 0;
        trainCtx.TransitionResource(m_LossBuffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        trainCtx.FillBuffer(m_LossBuffer, 0, zero, sizeof(uint32_t));

        uint32_t uvOffset = m_Model->GetMesh(0).attrib[ModelH3D::attrib_texcoord0].offset;
        uint32_t VertexStride = m_Model->GetVertexStride();

        trainCtx.SetRootSignature(m_GateTrainRootSig);
        trainCtx.SetDescriptorHeap(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, Renderer::s_TextureHeap.GetHeapPointer());
        
        float actualFeatureLR = m_GlobalLearningRate * (1.0f - m_LearningRateRatio);
        float actualMLPLR = m_GlobalLearningRate * m_LearningRateRatio * 0.05f;

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
            uint32_t meshColorResolution;
            uint32_t pointsPerTri;
        } cb = {
            m_TrainingStep, m_TotalTriangles, actualFeatureLR, actualMLPLR, m_AdamEpsilon,
            m_AdamBeta1, m_AdamBeta2, m_WeightDecay, m_ScreenSpaceRatio, VertexStride, uvOffset,
            (uint32_t)g_SceneColorBuffer.GetWidth(), (uint32_t)g_SceneColorBuffer.GetHeight(), m_Resolution, m_PointsPerTri
        };
        trainCtx.SetConstantArray(0, 15, &cb);

        // --- BACKPROP SETUP ---

        trainCtx.GetCommandList()->SetComputeRootShaderResourceView(1, m_GlobalTriangleBuffer.GetGpuVirtualAddress());
        trainCtx.GetCommandList()->SetComputeRootShaderResourceView(2, m_Model->GetVertexBuffer().BufferLocation);
        trainCtx.GetCommandList()->SetComputeRootShaderResourceView(12, m_VertexMappingBuffer.GetGpuVirtualAddress());
        trainCtx.GetCommandList()->SetComputeRootShaderResourceView(13, m_UniqueFeatureBuffer.GetGpuVirtualAddress());

        trainCtx.SetDynamicDescriptor(3, 0, visibilityBuffer.GetSRV());
        trainCtx.SetDescriptorTable(4, m_Model->GetSRVs(0));

        trainCtx.SetBufferUAV(5, m_UniqueFeatureBuffer);
        trainCtx.SetBufferUAV(6, m_GateFeatureGradientBuffer);
        trainCtx.SetBufferUAV(7, m_GateFeatureAdamBuffer);

        trainCtx.SetBufferUAV(8, m_GateMLPBuffer);
        trainCtx.SetBufferUAV(9, m_GateMLPGradientBuffer);
        trainCtx.SetBufferUAV(10, m_GateMLPAdamBuffer);
        trainCtx.SetBufferUAV(11, m_LossBuffer);

        // 1. Backprop
        trainCtx.SetPipelineState(m_GateBackpropPSO);
        trainCtx.Dispatch(m_BackpropDispatchedGroups, 1, 1);

        trainCtx.InsertUAVBarrier(m_GateFeatureGradientBuffer);
        trainCtx.InsertUAVBarrier(m_GateMLPGradientBuffer);

        // 2. Optimize MLP
        trainCtx.SetPipelineState(m_GateOptMLPPSO);
        trainCtx.Dispatch(1, 1, 1);

        // 3. Optimize features
        trainCtx.SetPipelineState(m_GateOptFeatPSO);
        trainCtx.Dispatch(Math::DivideByMultiple(m_UniqueSpatialVertexCount * 2, 1024), 1, 1);

        trainCtx.InsertUAVBarrier(m_UniqueFeatureBuffer);

		// 4. Broadcast features to duplicates
        trainCtx.SetPipelineState(m_GateBroadcastPSO);
        trainCtx.SetBufferUAV(5, m_GateFeatureBuffer);

        trainCtx.Dispatch(Math::DivideByMultiple(m_TotalTriangles * m_PointsPerTri, 1024), 1, 1);

        trainCtx.TransitionResource(m_GateFeatureBuffer, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        trainCtx.TransitionResource(m_GateMLPBuffer, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);

        uint32_t* mappedData = (uint32_t*)m_LossReadbackBuffer.Map();
        if (mappedData)
        {
            // Statické promìnné pro mìøení èasu a prùmìrování mezi framy
            static auto lastRecordTime = std::chrono::high_resolution_clock::now();
            static float accumulatedLoss = 0.0f;
            static uint32_t lossSamples = 0;

            // Spoèítáme loss pro TENTO konkrétní frame
            float totalLoss = (float)(mappedData[0]) / 1000.0f;
            float frameAverageLoss = totalLoss / (m_BackpropDispatchedGroups * 1024.0f);

            // Pøièteme do naší "èekárny"
            accumulatedLoss += frameAverageLoss;
            lossSamples++;

            // Zkontrolujeme, kolik èasu ubìhlo
            auto currentTime = std::chrono::high_resolution_clock::now();
            float elapsedTime = std::chrono::duration<float>(currentTime - lastRecordTime).count();

            if (elapsedTime >= 0.1f)
            {
                m_LossHistory[m_LossHistoryOffset] = accumulatedLoss / (float)lossSamples;
                m_LossHistoryOffset = (m_LossHistoryOffset + 1) % MAX_LOSS_HISTORY;

                // Resetujeme poèítadla pro další pùlsekundu
                accumulatedLoss = 0.0f;
                lossSamples = 0;
                lastRecordTime = currentTime;
            }

            m_LossReadbackBuffer.Unmap();
        }

        // 2. Až TEÏ zadáme GPU pøíkaz, a zkopíruje data z AKTUÁLNÍHO snímku pro pøíštì
        trainCtx.TransitionResource(m_LossBuffer, D3D12_RESOURCE_STATE_COPY_SOURCE);
        trainCtx.GetCommandList()->CopyResource(m_LossReadbackBuffer.GetResource(), m_LossBuffer.GetResource());

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

        gfxContext.SetBufferSRV(4, m_GlobalTriangleBuffer);

        uint32_t globalTriangleOffset = 0;
        for (uint32_t meshIndex = 0; meshIndex < m_Model->GetMeshCount(); ++meshIndex)
        {
            const ModelH3D::Mesh& mesh = m_Model->GetMesh(meshIndex);
            uint32_t indexCount = mesh.indexCount;
            uint32_t startIndex = mesh.indexDataByteOffset / sizeof(uint16_t);
            uint32_t baseVertex = mesh.vertexDataByteOffset / m_Model->GetVertexStride();

            // Global triangle offset in GatePS to acces the correct features
            uint32_t inferenceConstants[3] = { globalTriangleOffset, m_Resolution, m_PointsPerTri };
            gfxContext.SetConstantArray(3, 3, inferenceConstants);
            gfxContext.DrawIndexed(indexCount, startIndex, baseVertex);

            globalTriangleOffset += (indexCount / 3);
        }
    }

    void Gate::RenderGUI()
    {
        ImGui::Begin("GATE Training Configuration");
        
        ImGui::Spacing();
        ImGui::Text("Network Status");
        ImGui::Text("Training Step: %u", m_TrainingStep);
        if (ImGui::Button("Reset Training", ImVec2(ImGui::GetContentRegionAvail().x, 30)))
            ResetTraining();
        ImGui::Checkbox("Pause Training", &m_IsTrainingPaused);
        ImGui::Spacing();

        ImGui::Separator();
        ImGui::Spacing();
        ImGui::SliderInt("Backprop Steps", &m_BackpropDispatchedGroups, 1, 1024, "%d Groups * 1024 Threads");
        ImGui::SliderFloat("Screen Space Ratio", &m_ScreenSpaceRatio, 0.0f, 1.0f, "%.2f");
        ImGui::Spacing();

        ImGui::Separator();
        ImGui::Spacing();
        ImGui::SliderFloat("Learning Rate", &m_GlobalLearningRate, 0.00001f, 0.1f, "%.6f", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("Features/MLP Ratio", &m_LearningRateRatio, 0.0f, 1.0f, "%.2f");
        ImGui::Spacing();

        ImGui::Separator();
        ImGui::Spacing();
		ImGui::Text("Training Loss (MSE)"); // Mean Squared Error
        float currentLoss = m_LossHistory[(m_LossHistoryOffset == 0 ? MAX_LOSS_HISTORY : m_LossHistoryOffset) - 1];
        char overlay[32];
        sprintf_s(overlay, "Loss: %.5f", currentLoss);
        float maxLoss = *std::max_element(m_LossHistory.begin(), m_LossHistory.end());
        float graphMax = std::max(maxLoss * 1.2f, 0.001f);
        float graphHeight = 120.f;
        ImGui::PlotLines("##LossGraph", m_LossHistory.data(), MAX_LOSS_HISTORY, m_LossHistoryOffset, overlay,
            0.0f, graphMax, ImVec2(ImGui::GetContentRegionAvail().x, graphHeight));
        ImGui::Spacing();

        ImGui::End();
    }

    void Gate::ResetTraining()
    {
        m_TrainingStep = 1;
        AllocateBuffers();
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

        m_UniqueFeatureBuffer.Destroy();
        m_VertexMappingBuffer.Destroy();
    }
}