// Gate.cpp

#include "Gate.h"

#include <imgui/imgui.h>
#include <vector>
#include <unordered_map>
#include <DirectXMath.h>
#include <chrono>
#include <algorithm>
#include <ppl.h>
#include <numeric>

#include "Renderer.h"
#include "EngineTuning.h"

// Compiled Shaders
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

// --- External Dependencies ---
extern Microsoft::WRL::ComPtr<ID3D12Resource> g_bvh_topLevelAccelerationStructure;

// --- GPU Constant Buffer Layouts ---
namespace
{
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
        uint32_t totalMeshColorPoints;
        float aoRadius;
        uint32_t uniqueVertexCount;
        DirectX::XMFLOAT3 sunDirection;
        uint32_t featureFloats;
        uint32_t featureQuartets;
        uint32_t mlpQuartets;
        uint32_t learningMode;
        float maxGradientClip;
    };

    struct InferenceConstants
    {
        uint32_t globalTriangleOffset;
        uint32_t lightingMode;
        uint32_t renderFlags;
        uint32_t materialIdx;
        DirectX::XMFLOAT3 sunDirection;
        float sunIntensity;
        uint32_t featureFloats;
        uint32_t featureQuartets;
        float pad[2];
        DirectX::XMFLOAT3 cameraPos;
        float pad2;
    };
}

namespace Sponza
{
    // =========================================================================
    // Constructor & Destructor
    // =========================================================================

    Gate::Gate() :
        m_GatePSO(L"GATE: Forward PSO"),
        m_GateBackpropPSO(L"GATE: Backprop"),
        m_GateOptMLPPSO(L"GATE: Optimize MLP"),
        m_GateOptFeatPSO(L"GATE: Optimize Features"),
        m_EncodeColorPSO(L"GATE: Encode UVs CS"),
        m_Model(nullptr)
    {
    }

    Gate::~Gate()
    {
        Cleanup();
    }

    // =========================================================================
    // Lifecycle
    // =========================================================================

    void Gate::LoadModel(ModelH3D& model)
    {
        //model.Load(L"Sponza/sponza.h3d");
        model.Load(L"StanfordDragon/Dragon.h3d");
        //model.Load(L"Table/Table.h3d");
        m_Model = &model;
	}

    void Gate::Startup(DXGI_FORMAT colorFormat, DXGI_FORMAT depthFormat)
    {
        m_GateColorBuffer.Create(L"Gate Output Buffer", g_SceneColorBuffer.GetWidth(), g_SceneColorBuffer.GetHeight(), 1, g_SceneColorBuffer.GetFormat());
        m_VisColorBuffer.Create(L"Visibility Vis Buffer", g_SceneColorBuffer.GetWidth(), g_SceneColorBuffer.GetHeight(), 1, DXGI_FORMAT_R8G8B8A8_UNORM);

        m_LossHistory.resize(MAX_LOSS_HISTORY, 0.0f);
        m_LossBuffer.Create(L"Loss Buffer", 1, 4);
        m_LossReadbackBuffer.Create(L"Loss Readback", 1, 4);

        BuildSpatialIndex();
        AllocateBuffers();
        InitializePSOs(colorFormat, depthFormat);

        // --- Initialize HW GPU Timers ---
        D3D12_QUERY_HEAP_DESC queryHeapDesc = {};
        queryHeapDesc.Count = 10; // We need 10 timestamps
        queryHeapDesc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
        Graphics::g_Device->CreateQueryHeap(&queryHeapDesc, IID_PPV_ARGS(&m_GpuTimerHeap));

        auto heapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK);
        auto bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(10 * sizeof(uint64_t));
        Graphics::g_Device->CreateCommittedResource(
            &heapProps, D3D12_HEAP_FLAG_NONE, &bufferDesc,
            D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&m_GpuTimerReadback));

        // Get the GPU timestamp frequency (ticks per second) to convert to milliseconds
        Graphics::g_CommandManager.GetGraphicsQueue().GetCommandQueue()->GetTimestampFrequency(&m_GpuTimestampFreq);
    }

    void Gate::ResetTraining()
    {
        Graphics::g_CommandManager.IdleGPU();
        m_TrainingStep = 1;

        m_Config.resolution = m_Config.desiredResolution;
        m_Config.featureFloats = m_Config.desiredFeatureFloats;
        m_FeatureQuartets = (m_Config.featureFloats + 3) / 4;
        m_Config.useDeduplication = m_Config.desiredUseDeduplication;
        m_Config.useMaxEdgeLength = m_Config.desiredUseMaxEdgeLength;

        BuildSpatialIndex();
        AllocateBuffers();

        std::fill(m_LossHistory.begin(), m_LossHistory.end(), 0.0f);
        m_LossHistoryOffset = 0;
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

        m_UniqueFeatureBuffer.Destroy();
        m_VertexMappingBuffer.Destroy();

        m_GateFeatureDirtyBuffer.Destroy();
        m_UniqueToDuplicateOffsetBuffer.Destroy();
        m_UniqueToDuplicateCountBuffer.Destroy();
        m_DuplicateIndicesBuffer.Destroy();

        m_GateColorBuffer.Destroy();
        m_VisColorBuffer.Destroy();
        m_LossBuffer.Destroy();
        m_LossReadbackBuffer.Destroy();
    }

    // =========================================================================
    // Setup & Initialization Helpers
    // =========================================================================

    void Gate::BuildSpatialIndex()
    {
        m_Config.desiredResolution = m_Config.resolution;
        m_Config.desiredFeatureFloats = m_Config.featureFloats;
        m_Config.useMaxEdgeLength = m_Config.desiredUseMaxEdgeLength;
        m_Config.desiredUseDeduplication = m_Config.useDeduplication;

        auto tStart = std::chrono::high_resolution_clock::now();

        uint32_t vertexStride = m_Model->GetVertexStride();
        m_TotalVertices = m_Model->GetVertexBuffer().SizeInBytes / vertexStride;

        // 1. Calculate Edges
        std::vector<float> meshMaxEdges, meshAvgEdges;
        float globalMaxEdge = 0.0f, globalMaxAvgEdge = 0.0f;
        CalculateMeshEdgeLengths(meshMaxEdges, meshAvgEdges, globalMaxEdge, globalMaxAvgEdge);

        // 2. Generate Points
        std::vector<GlobalTriangle> globalTris;
        std::vector<Int3> allQuantizedPositions;
        std::vector<uint32_t> pointToMeshMap;
        GenerateQuantizedPoints(meshMaxEdges, meshAvgEdges, globalMaxEdge, globalMaxAvgEdge, globalTris, allQuantizedPositions, pointToMeshMap);

        // 3. Sort & Deduplicate
        std::vector<uint32_t> duplicateToUniqueMap(m_TotalMeshColorPoints);
        BuildInvertedSpatialIndex(allQuantizedPositions, pointToMeshMap, duplicateToUniqueMap);

        // 4. Create Buffers
        m_VertexMappingBuffer.Create(L"Mesh Colors Mapping Buffer", m_TotalMeshColorPoints, sizeof(uint32_t), duplicateToUniqueMap.data());
        m_GlobalTriangleBuffer.Create(L"Global Triangle Buffer", m_TotalTriangles, sizeof(GlobalTriangle), globalTris.data());

        auto tEnd = std::chrono::high_resolution_clock::now();
        m_CpuTimeBuildSpatialIndex = std::chrono::duration<float, std::milli>(tEnd - tStart).count();
    }

    void Gate::CalculateMeshEdgeLengths(std::vector<float>& outMeshMaxEdges, std::vector<float>& outMeshAvgEdges, float& outGlobalMaxEdge, float& outGlobalMaxAvgEdge)
    {
        uint32_t numMeshes = m_Model->GetMeshCount();
        uint32_t vertexStride = m_Model->GetVertexStride();
        const unsigned char* rawVertexData = m_Model->GetVertexData();
        const unsigned char* rawIndexData = m_Model->GetIndexData();

        outMeshMaxEdges.assign(numMeshes, 0.0f);
        outMeshAvgEdges.assign(numMeshes, 0.0f);
        outGlobalMaxEdge = 0.0f;
        outGlobalMaxAvgEdge = 0.0f;

        for (uint32_t i = 0; i < numMeshes; ++i)
        {
            const ModelH3D::Mesh& mesh = m_Model->GetMesh(i);
            uint32_t triCount = mesh.indexCount / 3;
            float maxEdge = 0.0f, totalEdge = 0.0f;
            uint32_t baseVertex = mesh.vertexDataByteOffset / vertexStride;
            const uint16_t* cpuIndexData = (const uint16_t*)(rawIndexData + mesh.indexDataByteOffset);

            for (uint32_t t = 0; t < mesh.indexCount; t += 3)
            {
                DirectX::XMVECTOR p0 = DirectX::XMLoadFloat3((DirectX::XMFLOAT3*)(rawVertexData + (cpuIndexData[t + 0] + baseVertex) * vertexStride));
                DirectX::XMVECTOR p1 = DirectX::XMLoadFloat3((DirectX::XMFLOAT3*)(rawVertexData + (cpuIndexData[t + 1] + baseVertex) * vertexStride));
                DirectX::XMVECTOR p2 = DirectX::XMLoadFloat3((DirectX::XMFLOAT3*)(rawVertexData + (cpuIndexData[t + 2] + baseVertex) * vertexStride));

                float e0 = DirectX::XMVectorGetX(DirectX::XMVector3Length(DirectX::XMVectorSubtract(p1, p0)));
                float e1 = DirectX::XMVectorGetX(DirectX::XMVector3Length(DirectX::XMVectorSubtract(p2, p1)));
                float e2 = DirectX::XMVectorGetX(DirectX::XMVector3Length(DirectX::XMVectorSubtract(p0, p2)));

                float maxTriEdge = std::max({ e0, e1, e2 });
                totalEdge += (e0 + e1 + e2);
                if (maxTriEdge > maxEdge)
                    maxEdge = maxTriEdge;
            }

            outMeshMaxEdges[i] = maxEdge;
            if (maxEdge > outGlobalMaxEdge)
                outGlobalMaxEdge = maxEdge;

            float avgEdge = totalEdge / (triCount * 3.0f);
            outMeshAvgEdges[i] = avgEdge;
            if (avgEdge > outGlobalMaxAvgEdge)
                outGlobalMaxAvgEdge = avgEdge;
        }
    }

    void Gate::GenerateQuantizedPoints(const std::vector<float>& meshMaxEdges, const std::vector<float>& meshAvgEdges, float globalMaxEdge, float globalMaxAvgEdge, std::vector<GlobalTriangle>& outGlobalTris, std::vector<Int3>& outQuantizedPositions, std::vector<uint32_t>& outPointToMeshMap)
    {
        uint32_t numMeshes = m_Model->GetMeshCount();
        uint32_t vertexStride = m_Model->GetVertexStride();
        const unsigned char* rawVertexData = m_Model->GetVertexData();
        const unsigned char* rawIndexData = m_Model->GetIndexData();

        std::vector<uint32_t> meshTriOffsets(numMeshes);
        m_TotalTriangles = 0;
        uint32_t currentGlobalPointOffset = 0;
        const uint32_t MAX_RES = m_Config.resolution;

        if (globalMaxEdge == 0.0f)
            globalMaxEdge = 1.0f;
        if (globalMaxAvgEdge == 0.0f)
            globalMaxAvgEdge = 1.0f;

        for (uint32_t i = 0; i < numMeshes; ++i)
        {
            meshTriOffsets[i] = m_TotalTriangles;
            const ModelH3D::Mesh& mesh = m_Model->GetMesh(i);
            uint32_t triCount = mesh.indexCount / 3;

            float edgeRatio = m_Config.useMaxEdgeLength ? (meshMaxEdges[i] / globalMaxEdge) : (meshAvgEdges[i] / globalMaxAvgEdge);
            uint32_t meshRes = std::max(1u, std::min(MAX_RES, (uint32_t)std::round((float)MAX_RES * edgeRatio)));
            uint32_t meshPtsPerTri = (meshRes + 1) * (meshRes + 2) / 2;

            for (uint32_t t = 0; t < triCount; ++t)
            {
                outGlobalTris.push_back({ 0, 0, 0, mesh.materialIndex, currentGlobalPointOffset, meshRes, meshPtsPerTri, 0 });
                currentGlobalPointOffset += meshPtsPerTri;
            }
            m_TotalTriangles += triCount;
        }
        m_TotalMeshColorPoints = currentGlobalPointOffset;

        outQuantizedPositions.resize(m_TotalMeshColorPoints);
        outPointToMeshMap.resize(m_TotalMeshColorPoints);

        std::vector<std::vector<DirectX::XMFLOAT3>> precomputedBarycentrics(MAX_RES + 1);
        for (uint32_t r = 1; r <= MAX_RES; ++r)
        {
            precomputedBarycentrics[r].resize((r + 1) * (r + 2) / 2);
            uint32_t idx = 0;
            for (uint32_t i = 0; i <= r; ++i)
                for (uint32_t j = 0; j <= r - i; ++j)
                    precomputedBarycentrics[r][idx++] = { (float)i / r, (float)j / r, (float)(r - i - j) / r };
        }

        const float QUANTIZATION_FACTOR = 10000.0f;
        concurrency::parallel_for(uint32_t(0), numMeshes, [&](uint32_t meshIndex)
            {
                const ModelH3D::Mesh& mesh = m_Model->GetMesh(meshIndex);
                uint32_t baseVertex = mesh.vertexDataByteOffset / vertexStride;
                const uint16_t* cpuIndexData = (const uint16_t*)(rawIndexData + mesh.indexDataByteOffset);
                uint32_t localTriOffset = meshTriOffsets[meshIndex];

                for (uint32_t i = 0; i < mesh.indexCount; i += 3)
                {
                    uint32_t i0 = cpuIndexData[i + 0] + baseVertex;
                    uint32_t i1 = cpuIndexData[i + 1] + baseVertex;
                    uint32_t i2 = cpuIndexData[i + 2] + baseVertex;

                    outGlobalTris[localTriOffset].i0 = i0;
                    outGlobalTris[localTriOffset].i1 = i1;
                    outGlobalTris[localTriOffset].i2 = i2;

                    uint32_t res = outGlobalTris[localTriOffset].resolution;
                    uint32_t pts = outGlobalTris[localTriOffset].pointsPerTri;
                    uint32_t pOffset = outGlobalTris[localTriOffset].pointOffset;

                    DirectX::XMFLOAT3* p0 = (DirectX::XMFLOAT3*)(rawVertexData + (i0 * vertexStride));
                    DirectX::XMFLOAT3* p1 = (DirectX::XMFLOAT3*)(rawVertexData + (i1 * vertexStride));
                    DirectX::XMFLOAT3* p2 = (DirectX::XMFLOAT3*)(rawVertexData + (i2 * vertexStride));

                    const auto& bary = precomputedBarycentrics[res];
                    for (uint32_t pt = 0; pt < pts; ++pt)
                    {
                        outQuantizedPositions[pOffset + pt] =
                        {
                            static_cast<int32_t>(std::round((bary[pt].x * p0->x + bary[pt].y * p1->x + bary[pt].z * p2->x) * QUANTIZATION_FACTOR)),
                            static_cast<int32_t>(std::round((bary[pt].x * p0->y + bary[pt].y * p1->y + bary[pt].z * p2->y) * QUANTIZATION_FACTOR)),
                            static_cast<int32_t>(std::round((bary[pt].x * p0->z + bary[pt].y * p1->z + bary[pt].z * p2->z) * QUANTIZATION_FACTOR))
                        };
                        outPointToMeshMap[pOffset + pt] = meshIndex;
                    }
                    localTriOffset++;
                }
            });
    }

    void Gate::BuildInvertedSpatialIndex(std::vector<Int3>& quantizedPositions, std::vector<uint32_t>& pointToMeshMap, std::vector<uint32_t>& outDuplicateToUniqueMap)
    {
        std::vector<uint32_t> sortIndices(m_TotalMeshColorPoints);
        std::iota(sortIndices.begin(), sortIndices.end(), 0);

        concurrency::parallel_sort(sortIndices.begin(), sortIndices.end(), [&](uint32_t a, uint32_t b) {
            if (!m_Config.useDeduplication)
            {
                if (pointToMeshMap[a] != pointToMeshMap[b])
                    return pointToMeshMap[a] < pointToMeshMap[b];
            }
            const Int3& posA = quantizedPositions[a];
            const Int3& posB = quantizedPositions[b];
            if (posA.x != posB.x) return posA.x < posB.x;
            if (posA.y != posB.y) return posA.y < posB.y;
            return posA.z < posB.z;
            });

        m_UniqueSpatialVertexCount = 0;
        if (m_TotalMeshColorPoints > 0)
        {
            outDuplicateToUniqueMap[sortIndices[0]] = 0;
            for (size_t i = 1; i < m_TotalMeshColorPoints; ++i)
            {
                uint32_t currIdx = sortIndices[i];
                uint32_t prevIdx = sortIndices[i - 1];

                const Int3& currPos = quantizedPositions[currIdx];
                const Int3& prevPos = quantizedPositions[prevIdx];
                bool different = (currPos.x != prevPos.x || currPos.y != prevPos.y || currPos.z != prevPos.z);

                if (!m_Config.useDeduplication && pointToMeshMap[currIdx] != pointToMeshMap[prevIdx])
                    different = true;

                if (different)
                    m_UniqueSpatialVertexCount++;
                outDuplicateToUniqueMap[currIdx] = m_UniqueSpatialVertexCount;
            }
            m_UniqueSpatialVertexCount++;
        }

        std::vector<uint32_t> uniqueCounts(m_UniqueSpatialVertexCount, 0);
        for (size_t i = 0; i < m_TotalMeshColorPoints; ++i)
            uniqueCounts[outDuplicateToUniqueMap[i]]++;

        std::vector<uint32_t> uniqueOffsets(m_UniqueSpatialVertexCount, 0);
        uint32_t offset = 0;
        for (size_t i = 0; i < m_UniqueSpatialVertexCount; ++i)
        {
            uniqueOffsets[i] = offset;
            offset += uniqueCounts[i];
        }

        std::vector<uint32_t> currentOffsets = uniqueOffsets;
        std::vector<uint32_t> duplicateIndices(m_TotalMeshColorPoints);
        for (size_t i = 0; i < m_TotalMeshColorPoints; ++i)
        {
            uint32_t uniqueID = outDuplicateToUniqueMap[i];
            duplicateIndices[currentOffsets[uniqueID]++] = i;
        }

        m_UniqueToDuplicateCountBuffer.Create(L"Unique To Duplicate Count", m_UniqueSpatialVertexCount, sizeof(uint32_t), uniqueCounts.data());
        m_UniqueToDuplicateOffsetBuffer.Create(L"Unique To Duplicate Offset", m_UniqueSpatialVertexCount, sizeof(uint32_t), uniqueOffsets.data());
        m_DuplicateIndicesBuffer.Create(L"Duplicate Indices", m_TotalMeshColorPoints, sizeof(uint32_t), duplicateIndices.data());
    }

    void Gate::AllocateBuffers()
    {
        srand(1337);

        std::vector<uint32_t> zeroDirty(m_UniqueSpatialVertexCount, 0);
        m_GateFeatureDirtyBuffer.Create(L"Feature Dirty Buffer", m_UniqueSpatialVertexCount, sizeof(uint32_t), zeroDirty.data());

        // A. DUPLICATED FEATURE BUFFER
        uint32_t totalFeatureFloats = m_TotalMeshColorPoints * m_FeatureQuartets;
        std::vector<DirectX::XMFLOAT4> duplicatedFeatures(totalFeatureFloats);
        for (uint32_t i = 0; i < totalFeatureFloats; ++i)
            duplicatedFeatures[i] = DirectX::XMFLOAT4((float)rand() / RAND_MAX, (float)rand() / RAND_MAX, (float)rand() / RAND_MAX, (float)rand() / RAND_MAX);
        m_GateFeatureBuffer.Create(L"DUPLICATED Feature Buffer", totalFeatureFloats, sizeof(DirectX::XMFLOAT4), duplicatedFeatures.data());

        // B. UNIQUE FEATURE BUFFER
        uint32_t uniqueFeatureFloats = m_UniqueSpatialVertexCount * m_FeatureQuartets;
        std::vector<DirectX::XMFLOAT4> uniqueFeatures(uniqueFeatureFloats);
        for (uint32_t i = 0; i < uniqueFeatureFloats; ++i)
            uniqueFeatures[i] = DirectX::XMFLOAT4((float)rand() / RAND_MAX, (float)rand() / RAND_MAX, (float)rand() / RAND_MAX, (float)rand() / RAND_MAX);
        m_UniqueFeatureBuffer.Create(L"UNIQUE Feature Buffer", uniqueFeatureFloats, sizeof(DirectX::XMFLOAT4), uniqueFeatures.data());

        std::vector<AdamData> initialFeatureAdam(uniqueFeatureFloats, { {0,0,0,0}, {0,0,0,0}, 0, {0,0,0} });
        m_GateFeatureAdamBuffer.Create(L"UNIQUE Feature Adam Buffer", uniqueFeatureFloats, sizeof(AdamData), initialFeatureAdam.data());
        std::vector<DirectX::XMINT4> zeroFeatureGradients(uniqueFeatureFloats, { 0, 0, 0, 0 });
        m_GateFeatureGradientBuffer.Create(L"UNIQUE Feature Gradients", uniqueFeatureFloats, sizeof(DirectX::XMINT4), zeroFeatureGradients.data());

        // C. MLP PARAMETERS (Dynamic Calculation)
        // Architecture: 
        // Layer 1: (Input Features -> 16 hidden nodes) + 16 biases
        // Layer 2: (16 hidden nodes -> 4 output nodes) + 4 biases = 68 parameters
        uint32_t layer1Params = (16 * (m_FeatureQuartets * 4)) + 16;
        uint32_t layer2Params = 68;

        m_MlpParameterCount = layer1Params + layer2Params;
        m_MlpQuartets = m_MlpParameterCount / 4;

        std::vector<float> initialWeights(m_MlpParameterCount);
        for (uint32_t i = 0; i < m_MlpParameterCount; ++i)
            initialWeights[i] = ((float)rand() / (float)RAND_MAX) * 0.2f - 0.1f;
        m_GateMLPBuffer.Create(L"MLP Parameters", m_MlpQuartets, sizeof(DirectX::XMFLOAT4), initialWeights.data());
        std::vector<DirectX::XMINT4> zeroMlpGradients(m_MlpQuartets, { 0, 0, 0, 0 });
        m_GateMLPGradientBuffer.Create(L"MLP Gradients", m_MlpQuartets, sizeof(DirectX::XMINT4), zeroMlpGradients.data());

        std::vector<AdamData> initialMLPAdam(m_MlpQuartets, { {0,0,0,0}, {0,0,0,0}, 0, {0,0,0} });
        m_GateMLPAdamBuffer.Create(L"MLP Adam Buffer", m_MlpQuartets, sizeof(AdamData), initialMLPAdam.data());
    }

    void Gate::InitializePSOs(DXGI_FORMAT colorFormat, DXGI_FORMAT depthFormat)
    {
        // 1. Setup Inference PSO
        m_GateRootSig.Reset(6, 1);
        m_GateRootSig[0].InitAsConstantBuffer(0); // b0
        m_GateRootSig[1].InitAsBufferSRV(0);      // t0 (FeatureBuffer)
        m_GateRootSig[2].InitAsBufferSRV(1);      // t1 (MLP)
        m_GateRootSig[3].InitAsConstants(1, 16);   // b1 (Inference Constants)
        m_GateRootSig[4].InitAsBufferSRV(2);      // t2 (GlobalTriangleBuffer)
        m_GateRootSig[5].InitAsDescriptorTable(1);
        m_GateRootSig[5].SetTableRange(0, D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, (UINT)-1, 1);
        m_GateRootSig.InitStaticSampler(0, Graphics::SamplerLinearWrapDesc);
        m_GateRootSig.Finalize(L"Gate Inference Root Sig", D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

        D3D12_INPUT_ELEMENT_DESC vertElem[] =
        {
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
        m_GateTrainRootSig.Reset(19, 1);
        m_GateTrainRootSig[0].InitAsConstants(0, 24); // register(b0)
        m_GateTrainRootSig[1].InitAsBufferSRV(0);     // TriangleBuffer register(t0)
        m_GateTrainRootSig[2].InitAsBufferSRV(1);     // VertexUVBuffer register(t1)

        m_GateTrainRootSig[3].InitAsDescriptorTable(1);
        m_GateTrainRootSig[3].SetTableRange(0, D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 2, 1, 0); // VisBuffer (t2)

        m_GateTrainRootSig[4].InitAsDescriptorTable(1);
        m_GateTrainRootSig[4].SetTableRange(0, D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, (UINT)-1, 1); // Bindless

        for (UINT i = 0; i < 8; ++i)
            m_GateTrainRootSig[5 + i].InitAsBufferUAV(i);

        for (UINT i = 0; i < 6; ++i)
            m_GateTrainRootSig[13 + i].InitAsBufferSRV(3 + i);

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

    // =========================================================================
    // Core Execution
    // =========================================================================

    void Gate::Train(ComputeContext& trainCtx, ColorBuffer& visibilityBuffer, Math::Vector3 sunDirection)
    {
        if (m_Config.isTrainingPaused)
            return;

        uint32_t zero = 0;
        trainCtx.TransitionResource(m_LossBuffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        trainCtx.FillBuffer(m_LossBuffer, 0, zero, sizeof(uint32_t));

        uint32_t uvOffset = m_Model->GetMesh(0).attrib[ModelH3D::attrib_texcoord0].offset;
        uint32_t VertexStride = m_Model->GetVertexStride();

        trainCtx.SetRootSignature(m_GateTrainRootSig);
        trainCtx.SetDescriptorHeap(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, Renderer::s_TextureHeap.GetHeapPointer());

        float actualFeatureLR = m_Config.globalLearningRate * m_Config.learningRateRatio;
        float actualMLPLR = m_Config.globalLearningRate * (1.0f - m_Config.learningRateRatio) * 0.05f;

        TrainingConstants cb = {
            m_TrainingStep, m_TotalTriangles, actualFeatureLR, actualMLPLR, m_Config.adamEpsilon,
            m_Config.adamBeta1, m_Config.adamBeta2, m_Config.weightDecay, m_Config.screenSpaceRatio, VertexStride, uvOffset,
            (uint32_t)g_SceneColorBuffer.GetWidth(), (uint32_t)g_SceneColorBuffer.GetHeight(),
            m_TotalMeshColorPoints, m_Config.aoRadius, m_UniqueSpatialVertexCount,
            DirectX::XMFLOAT3(sunDirection.GetX(), sunDirection.GetY(), sunDirection.GetZ()),
            m_Config.featureFloats, m_FeatureQuartets, m_MlpQuartets, (uint32_t)m_Config.learningMode, m_Config.maxGradientClip
        };
        trainCtx.SetConstantArray(0, sizeof(TrainingConstants) / 4, &cb);
        trainCtx.SetConstantArray(0, sizeof(TrainingConstants) / 4, &cb);

        // --- BACKPROP SETUP ---
        trainCtx.GetCommandList()->SetComputeRootShaderResourceView(1, m_GlobalTriangleBuffer.GetGpuVirtualAddress());
        trainCtx.GetCommandList()->SetComputeRootShaderResourceView(2, m_Model->GetVertexBuffer().BufferLocation);
        trainCtx.GetCommandList()->SetComputeRootShaderResourceView(13, m_VertexMappingBuffer.GetGpuVirtualAddress());
        trainCtx.GetCommandList()->SetComputeRootShaderResourceView(14, m_UniqueFeatureBuffer.GetGpuVirtualAddress());
        trainCtx.GetCommandList()->SetComputeRootShaderResourceView(15, g_bvh_topLevelAccelerationStructure->GetGPUVirtualAddress());

        trainCtx.SetDynamicDescriptor(3, 0, visibilityBuffer.GetSRV());
        trainCtx.SetDescriptorTable(4, m_Model->GetSRVs(0));

        trainCtx.SetBufferUAV(5, m_UniqueFeatureBuffer);
        trainCtx.SetBufferUAV(6, m_GateFeatureGradientBuffer);
        trainCtx.SetBufferUAV(7, m_GateFeatureAdamBuffer);

        trainCtx.SetBufferUAV(8, m_GateMLPBuffer);
        trainCtx.SetBufferUAV(9, m_GateMLPGradientBuffer);
        trainCtx.SetBufferUAV(10, m_GateMLPAdamBuffer);
        trainCtx.SetBufferUAV(11, m_LossBuffer);

        trainCtx.SetBufferUAV(12, m_GateFeatureDirtyBuffer);
        trainCtx.GetCommandList()->SetComputeRootShaderResourceView(16, m_UniqueToDuplicateOffsetBuffer.GetGpuVirtualAddress());
        trainCtx.GetCommandList()->SetComputeRootShaderResourceView(17, m_UniqueToDuplicateCountBuffer.GetGpuVirtualAddress());
        trainCtx.GetCommandList()->SetComputeRootShaderResourceView(18, m_DuplicateIndicesBuffer.GetGpuVirtualAddress());

        auto cmdList = trainCtx.GetCommandList();

        // 1. Backprop
        cmdList->EndQuery(m_GpuTimerHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0); // START 0
        trainCtx.SetPipelineState(m_GateBackpropPSO);
        trainCtx.Dispatch(m_Config.backpropDispatchedGroups, 1, 1);
        cmdList->EndQuery(m_GpuTimerHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 1); // END 1

        trainCtx.InsertUAVBarrier(m_GateFeatureGradientBuffer);
        trainCtx.InsertUAVBarrier(m_GateMLPGradientBuffer);

        // 2. Optimize MLP
        cmdList->EndQuery(m_GpuTimerHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 2); // START 2
        trainCtx.SetPipelineState(m_GateOptMLPPSO);
        trainCtx.Dispatch(1, 1, 1);
        cmdList->EndQuery(m_GpuTimerHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 3); // END 3

        // 3. Optimize features
        cmdList->EndQuery(m_GpuTimerHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 4); // START 4
        trainCtx.SetPipelineState(m_GateOptFeatPSO);
        trainCtx.Dispatch(Math::DivideByMultiple(m_UniqueSpatialVertexCount * m_FeatureQuartets, 1024), 1, 1);
        cmdList->EndQuery(m_GpuTimerHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 5); // END 5

        trainCtx.InsertUAVBarrier(m_UniqueFeatureBuffer);
        trainCtx.InsertUAVBarrier(m_GateFeatureDirtyBuffer);

        // 4. Broadcast
        cmdList->EndQuery(m_GpuTimerHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 6);
        trainCtx.SetPipelineState(m_GateBroadcastPSO);
        trainCtx.SetBufferUAV(5, m_GateFeatureBuffer);
        trainCtx.Dispatch(Math::DivideByMultiple(m_UniqueSpatialVertexCount, 1024), 1, 1);
        cmdList->EndQuery(m_GpuTimerHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 7);

        trainCtx.TransitionResource(m_GateFeatureBuffer, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        trainCtx.TransitionResource(m_GateMLPBuffer, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);

        uint32_t* mappedData = (uint32_t*)m_LossReadbackBuffer.Map();
        if (mappedData)
        {
            float totalLoss = (float)(mappedData[0]) / 1000.0f;
            float frameAverageLoss = totalLoss / (m_Config.backpropDispatchedGroups * 1024.0f);

            UpdateLossHistory(frameAverageLoss);

            m_LossReadbackBuffer.Unmap();
        }

        trainCtx.TransitionResource(m_LossBuffer, D3D12_RESOURCE_STATE_COPY_SOURCE);
        trainCtx.GetCommandList()->CopyResource(m_LossReadbackBuffer.GetResource(), m_LossBuffer.GetResource());

        m_TrainingStep++;
    }

    void Gate::RenderVisualization(GraphicsContext& gfxContext, const Camera& camera, DepthBuffer& depthBuffer,
        const D3D12_VIEWPORT& viewport, const D3D12_RECT& scissor, ColorBuffer& visibilityBuffer,
        Math::Vector3 sunDirection, float sunIntensity)
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
        gfxContext.SetDescriptorTable(5, m_Model->GetSRVs(0));

        auto cmdList = gfxContext.GetCommandList();
        cmdList->EndQuery(m_GpuTimerHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 8); // START 8

        // 1. Calculate static frame properties ONCE outside the loop
        uint32_t flags = 0;
        if (m_Config.texturesEnabled)         flags |= (1 << 0);
        if (m_Config.directionalLightEnabled) flags |= (1 << 1);

        InferenceConstants cb = {};
        cb.lightingMode = static_cast<uint32_t>(m_Config.lightingMode);
        cb.renderFlags = flags;
        cb.sunDirection = DirectX::XMFLOAT3(sunDirection.GetX(), sunDirection.GetY(), sunDirection.GetZ());
        cb.sunIntensity = sunIntensity;
        cb.featureFloats = m_Config.featureFloats;
        cb.featureQuartets = m_FeatureQuartets;
        cb.cameraPos = DirectX::XMFLOAT3(camera.GetPosition().GetX(), camera.GetPosition().GetY(), camera.GetPosition().GetZ());

        // 2. Loop through meshes
        uint32_t globalTriangleOffset = 0;
        for (uint32_t meshIndex = 0; meshIndex < m_Model->GetMeshCount(); ++meshIndex)
        {
            const ModelH3D::Mesh& mesh = m_Model->GetMesh(meshIndex);
            uint32_t indexCount = mesh.indexCount;
            uint32_t startIndex = mesh.indexDataByteOffset / sizeof(uint16_t);
            uint32_t baseVertex = mesh.vertexDataByteOffset / m_Model->GetVertexStride();

            // 3. Update ONLY what changes per mesh
            cb.globalTriangleOffset = globalTriangleOffset;
            cb.materialIdx = mesh.materialIndex;

            gfxContext.SetConstantArray(3, sizeof(InferenceConstants) / 4, &cb);
            gfxContext.DrawIndexed(indexCount, startIndex, baseVertex);

            globalTriangleOffset += (indexCount / 3);
        }

        cmdList->EndQuery(m_GpuTimerHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 9); // END 9
        cmdList->ResolveQueryData(m_GpuTimerHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 10, m_GpuTimerReadback.Get(), 0);
    }

    // =========================================================================
    // UI & Profiling Helpers
    // =========================================================================

    void Gate::UpdateLossHistory(float frameAverageLoss)
    {
        static auto lastRecordTime = std::chrono::high_resolution_clock::now();
        static float accumulatedLoss = 0.0f;
        static uint32_t lossSamples = 0;

        accumulatedLoss += frameAverageLoss;
        lossSamples++;

        auto currentTime = std::chrono::high_resolution_clock::now();
        float elapsedTime = std::chrono::duration<float>(currentTime - lastRecordTime).count();

        // Update the graph every 100ms
        if (elapsedTime >= 0.1f)
        {
            m_LossHistory[m_LossHistoryOffset] = accumulatedLoss / (float)lossSamples;
            m_LossHistoryOffset = (m_LossHistoryOffset + 1) % MAX_LOSS_HISTORY;

            accumulatedLoss = 0.0f;
            lossSamples = 0;
            lastRecordTime = currentTime;
        }
    }

    void Gate::ReadbackGpuTimers()
    {
        uint64_t* timestamps = nullptr;
        if (m_GpuTimerReadback && SUCCEEDED(m_GpuTimerReadback->Map(0, nullptr, (void**)&timestamps)))
        {
            if (m_GpuTimestampFreq > 0)
            {
                double invFreq = 1000.0 / (double)m_GpuTimestampFreq; // ms

                auto updateTimer = [&](float& trackingVar, int startIndex)
                    {
                        if (timestamps[startIndex + 1] > timestamps[startIndex])
                            trackingVar = trackingVar * 0.9f + (float)((timestamps[startIndex + 1] - timestamps[startIndex]) * invFreq) * 0.1f;
                    };

                updateTimer(m_GpuTimeBackprop, 0);
                updateTimer(m_GpuTimeOptMLP, 2);
                updateTimer(m_GpuTimeOptFeat, 4);
                updateTimer(m_GpuTimeBroadcast, 6);
                updateTimer(m_GpuTimeRender, 8);
            }
            m_GpuTimerReadback->Unmap(0, nullptr);
        }
    }

    void Gate::RenderGUI()
    {
        ReadbackGpuTimers();

        // --- RENDER IMGUI ---
        ImGui::Begin("GATE Training Configuration");

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        ImGui::Text("Performance Timings (ms)");
        ImGui::Text("Build Spatial Index (CPU): %.2f ms", m_CpuTimeBuildSpatialIndex);
        ImGui::Spacing();
        ImGui::TextDisabled("--- Real GPU Execution Time ---");
        ImGui::Text("Backprop:      %.4f ms", m_GpuTimeBackprop);
        ImGui::Text("Optimize MLP:  %.4f ms", m_GpuTimeOptMLP);
        ImGui::Text("Optimize Feat: %.4f ms", m_GpuTimeOptFeat);
        ImGui::Text("Broadcast:     %.4f ms", m_GpuTimeBroadcast);
        ImGui::Text("Forward Render:%.4f ms", m_GpuTimeRender);
        ImGui::Spacing();

        ImGui::Separator();
        ImGui::Spacing();
        ImGui::Text("Training Loss (MSE)");
        float currentLoss = m_LossHistory[(m_LossHistoryOffset == 0 ? MAX_LOSS_HISTORY : m_LossHistoryOffset) - 1];
        char overlay[32];
        sprintf_s(overlay, "Loss: %.5f", currentLoss);
        float maxLoss = *std::max_element(m_LossHistory.begin(), m_LossHistory.end());
        float graphMax = std::max(maxLoss * 1.2f, 0.001f);
        float graphHeight = 120.f;
        ImGui::PlotLines("##LossGraph", m_LossHistory.data(), MAX_LOSS_HISTORY, m_LossHistoryOffset, overlay,
            0.0f, graphMax, ImVec2(ImGui::GetContentRegionAvail().x, graphHeight));
        ImGui::Spacing();

        ImGui::Separator();
        ImGui::Spacing();
        ImGui::Text("Network Status");
        ImGui::Text("Training Step: %u", m_TrainingStep);
        ImGui::InputInt("Max Resolution Scale", &m_Config.desiredResolution, 1);
        m_Config.desiredResolution = std::max(1, std::min(m_Config.desiredResolution, 1024));
        ImGui::SliderInt("Feature Dimension", (int*)&m_Config.desiredFeatureFloats, 1, 32);
        ImGui::Checkbox("Use Max Edge Length (Off = Average)", &m_Config.desiredUseMaxEdgeLength);
        ImGui::Checkbox("Use Spatial Deduplication", &m_Config.desiredUseDeduplication);

        if (m_Config.desiredResolution != (int)m_Config.resolution ||
            m_Config.desiredFeatureFloats != m_Config.featureFloats ||
            m_Config.desiredUseMaxEdgeLength != m_Config.useMaxEdgeLength ||
            m_Config.desiredUseDeduplication != m_Config.useDeduplication)
            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f), "Architecture changed! Reset training to apply.");

        if (ImGui::Button("Reset Training & Apply", ImVec2(ImGui::GetContentRegionAvail().x, 30)))
            ResetTraining();

        ImGui::Checkbox("Pause Training", &m_Config.isTrainingPaused);

        ImGui::Separator();
        ImGui::Spacing();
        ImGui::SliderInt("Backprop Steps", &m_Config.backpropDispatchedGroups, 1, 1024, "%d Groups * 1024 Threads");
        ImGui::SliderFloat("Screen Space Ratio", &m_Config.screenSpaceRatio, 0.0f, 1.0f, "%.2f");
        ImGui::SliderFloat("AO Radius", &m_Config.aoRadius, 10.0f, 1000.0f, "%.1f");

        // --- SMART UI LOGIC ---
        const char* learningModes[] = { "Learn AO + Shadows", "Learn AO Only", "Learn Shadows Only", "Learn Color (RGB Test)" };
        if (ImGui::Combo("Learning Target", &m_Config.learningMode, learningModes, IM_ARRAYSIZE(learningModes)))
        {
            if (m_Config.learningMode == 1 && (m_Config.lightingMode == 2 || m_Config.lightingMode == 3)) m_Config.lightingMode = 1;
            else if (m_Config.learningMode == 2 && (m_Config.lightingMode == 1 || m_Config.lightingMode == 3)) m_Config.lightingMode = 2;
            else if (m_Config.learningMode == 3) m_Config.lightingMode = 5;
        }

        const char* lightingModes[] = { "No Shadows/AO", "AO Only", "Shadows Only", "AO + Shadows", "Debug: Subdivision Grid", "Network RGB" };
        if (ImGui::BeginCombo("Viewing Mode", lightingModes[m_Config.lightingMode]))
        {
            for (int i = 0; i < 6; i++)
            {
                bool isValid = true;
                if (m_Config.learningMode == 1 && (i == 2 || i == 3 || i == 5)) isValid = false;
                if (m_Config.learningMode == 2 && (i == 1 || i == 3 || i == 5)) isValid = false;
                if (m_Config.learningMode == 3 && (i >= 0 && i <= 3)) isValid = false;

                if (isValid)
                {
                    bool isSelected = (m_Config.lightingMode == i);
                    if (ImGui::Selectable(lightingModes[i], isSelected))
                        m_Config.lightingMode = i;

                    if (isSelected) ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }

        if (m_Config.learningMode != 0)
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "* Viewing mode restricted to active learning target.");

        ImGui::Checkbox("Textures Enabled", &m_Config.texturesEnabled);
        ImGui::Checkbox("Directional Light Enabled", &m_Config.directionalLightEnabled);
        ImGui::Spacing();

        ImGui::Separator();
        ImGui::Spacing();
        ImGui::SliderFloat("Learning Rate", &m_Config.globalLearningRate, 0.0001f, 0.1f, "%.6f", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("Features/MLP Ratio", &m_Config.learningRateRatio, 0.0f, 1.0f, "%.2f");
        ImGui::SliderFloat("Max Gradient Clip", &m_Config.maxGradientClip, 0.0001f, 0.1f, "%.4f", ImGuiSliderFlags_Logarithmic);
        ImGui::Spacing();

        ImGui::End();
    }
}