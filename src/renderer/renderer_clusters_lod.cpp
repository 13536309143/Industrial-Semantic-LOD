//==============================================================================
// src/renderer/renderer_clusters_lod.cpp
// Implements the clustered LOD raster renderer.
// The renderer builds traversal, setup, render, and optional sort pipelines that select visible clusters before mesh shading.
//==============================================================================



#include <volk.h>
#include <nvutils/alignment.hpp>
#include <fmt/format.h>
#include <algorithm>
#include <cmath>
#include "renderer.hpp"
#include "shaderio.h"
#include "adaptive_two_pass.hpp"
#include "adaptive_cost.hpp"

namespace {
float matrixMaxAbsDelta(const glm::mat4& a, const glm::mat4& b)
{
  float maxDelta = 0.0f;
  for(int col = 0; col < 4; col++)
  {
    for(int row = 0; row < 4; row++)
    {
      maxDelta = std::max(maxDelta, std::abs(a[col][row] - b[col][row]));
    }
  }
  return maxDelta;
}
}


namespace lodclusters {


// Renderer implementation that drives setup, traversal, optional sorting, and cluster mesh shading.
class RendererRasterClustersLod : public Renderer
{
public:

  virtual bool init(Resources& res, RenderScene& rscene, const RendererConfig& config) override;
  virtual void render(VkCommandBuffer primary, Resources& res, RenderScene& rscene, const FrameConfig& frame, nvvk::ProfilerGpuTimer& profiler) override;
  virtual void updatedFrameBuffer(Resources& res, RenderScene& rscene) override;
  virtual void deinit(Resources& res) override;


private:


  bool initShaders(Resources& res, RenderScene& scene, const RendererConfig& config);


  struct Shaders
  {
    shaderc::SpvCompilationResult graphicsMesh;
    shaderc::SpvCompilationResult graphicsFragment;
    shaderc::SpvCompilationResult computeTraversalPresort;
    shaderc::SpvCompilationResult computeAssemblyVisibility;
    shaderc::SpvCompilationResult computeAssemblyCandidates;
    shaderc::SpvCompilationResult computeTraversalInit;
    shaderc::SpvCompilationResult computeTraversalRun;
    shaderc::SpvCompilationResult computeTraversalGroups;
    shaderc::SpvCompilationResult computeBuildSetup;
  };


  struct Pipelines
  {
    VkPipeline graphicsMesh            = nullptr;
    VkPipeline graphicsBboxes          = nullptr;
    VkPipeline computeTraversalPresort = nullptr;
    VkPipeline computeAssemblyVisibility = nullptr;
    VkPipeline computeAssemblyCandidates = nullptr;
    VkPipeline computeTraversalInit    = nullptr;
    VkPipeline computeTraversalRun     = nullptr;
    VkPipeline computeTraversalGroups  = nullptr;
    VkPipeline computeBuildSetup       = nullptr;
  };
  Shaders            m_shaders;
  Pipelines          m_pipelines;
  VkShaderStageFlags m_stageFlags{};
  VkPipelineLayout   m_pipelineLayout{};
  nvvk::DescriptorPack m_dsetPack;
  nvvk::Buffer m_sceneBuildBuffer;
  nvvk::Buffer m_sceneTraversalBuffer;
  nvvk::Buffer m_sceneDataBuffer;
  nvvk::Buffer m_assemblyNodeBuffer;
  nvvk::Buffer m_assemblyStateBuffer;
  nvvk::Buffer m_candidateRanges, m_candidateMembers, m_candidateInstances, m_candidateControl;
  shaderio::SceneBuilding m_sceneBuildShaderio;
  AdaptiveTwoPassController m_adaptiveTwoPass;
  AdaptiveCostController m_adaptiveCost;
  VkQueryPool m_costQueries = VK_NULL_HANDLE;
  double m_timestampPeriod = 0;
  struct CostSample { bool pending=false; bool dual=false; bool eligible=false; };
  std::vector<CostSample> m_costSamples;
  bool m_lastCostValid=false,m_lastCostDual=false;
  uint64_t m_lastCostFrame=0;
};


bool RendererRasterClustersLod::initShaders(Resources& res, RenderScene& rscene, const RendererConfig& config)
{

  if(!initBasicShaders(res, rscene, config))
  {
    return false;
  }


  shaderc::CompileOptions options = res.makeCompilerOptions();


  uint32_t meshletTriangles = shaderio::adjustClusterProperty(rscene.scene->m_maxClusterTriangles);

  uint32_t meshletVertices  = shaderio::adjustClusterProperty(rscene.scene->m_maxClusterVertices);

  LOGI("mesh shader config: %d triangles %d vertices\n", meshletTriangles, meshletVertices);


  options.SetOptimizationLevel(shaderc_optimization_level_performance);
  options.AddMacroDefinition("PARALLEL_GROUPS", config.useParallelGroups ? "1" : "0");
  options.AddMacroDefinition("QUALITY_BOUNDED_ASSEMBLY", config.boundAssemblyLod ? "1" : "0");
  options.AddMacroDefinition("USE_COMPACT_CANDIDATES",config.useAssemblyCompact && config.useAssemblyPrefilter && !config.useSorting ? "1":"0");
  options.AddMacroDefinition("USE_ASSEMBLY_PREFILTER", config.useAssemblyPrefilter ? "1" : "0");
  options.AddMacroDefinition("USE_ASSEMBLY_EARLY_EXIT", config.useAssemblyEarlyExit ? "1" : "0");
  options.AddMacroDefinition("USE_LOW_DETAIL_FAST_PATH", config.useLowDetailFastPath ? "1" : "0");
  options.AddMacroDefinition("USE_HIZ_OCCLUSION", config.useHizOcclusion ? "1" : "0");
  options.AddMacroDefinition("GROUP_CLUSTER_LANES",fmt::format("{}",std::clamp(config.groupClusterLanes,1u,32u)));



  options.AddMacroDefinition("SUBGROUP_SIZE", fmt::format("{}", res.m_physicalDeviceInfo.properties11.subgroupSize));
  options.AddMacroDefinition("USE_16BIT_DISPATCH", fmt::format("{}", res.m_use16bitDispatch ? 1 : 0));
  options.AddMacroDefinition("CLUSTER_VERTEX_COUNT", fmt::format("{}", meshletVertices));
  options.AddMacroDefinition("CLUSTER_TRIANGLE_COUNT", fmt::format("{}", meshletTriangles));

  options.AddMacroDefinition("TARGETS_RASTERIZATION", "1");
  options.AddMacroDefinition("ADAPTIVE_TEMPORAL_GUARD", "1");

  options.AddMacroDefinition("USE_STREAMING", rscene.useStreaming ? "1" : "0");

  options.AddMacroDefinition("USE_SORTING", config.useSorting ? "1" : "0");

  options.AddMacroDefinition("USE_CULLING", config.useCulling ? "1" : "0");


  options.AddMacroDefinition("USE_TWO_PASS_CULLING", config.useCulling && config.useTwoPassCulling ? "1" : "0");

  options.AddMacroDefinition("USE_RENDER_STATS", config.useRenderStats ? "1" : "0");

  options.AddMacroDefinition("USE_SEPARATE_GROUPS", config.useSeparateGroups ? "1" : "0");
  options.AddMacroDefinition("USE_EXT_MESH_SHADER", fmt::format("{}", config.useEXTmeshShader ? 1 : 0));
  options.AddMacroDefinition("MESHSHADER_WORKGROUP_SIZE", fmt::format("{}", m_meshShaderWorkgroupSize));
  options.AddMacroDefinition("MESHSHADER_BBOX_COUNT", fmt::format("{}", m_meshShaderBoxes));


  options.AddMacroDefinition("ALLOW_VERTEX_NORMALS", rscene.scene->m_hasVertexNormals && res.m_supportsBarycentrics ? "1" : "0");

  options.AddMacroDefinition("ALLOW_VERTEX_TANGENTS", rscene.scene->m_hasVertexTangents && res.m_supportsBarycentrics ? "1" : "0");

  options.AddMacroDefinition("ALLOW_VERTEX_TEXCOORDS", rscene.scene->m_hasVertexTexCoord0 ? "1" : "0");


  options.AddMacroDefinition("ALLOW_SHADING", config.useShading ? "1" : "0");


  options.AddMacroDefinition("USE_DEPTH_ONLY", !config.useShading && config.useDepthOnly ? "1" : "0");

  options.AddMacroDefinition("DEBUG_VISUALIZATION", config.useDebugVisualization && res.m_supportsBarycentrics ? "1" : "0");

  options.AddMacroDefinition("USE_TWO_SIDED", rscene.scene->m_hasTwoSided && !config.forceTwoSided ? "1" : "0");

  options.AddMacroDefinition("USE_FORCED_TWO_SIDED", config.forceTwoSided ? "1" : "0");

  options.AddMacroDefinition("USE_FORCED_INVISIBLE_CULLING", "0");


    res.compileShader(m_shaders.graphicsMesh, VK_SHADER_STAGE_MESH_BIT_NV, "render/clusters.mesh.glsl", &options);

    res.compileShader(m_shaders.graphicsFragment, VK_SHADER_STAGE_FRAGMENT_BIT, "render/frag.glsl", &options);

    res.compileShader(m_shaders.computeTraversalPresort, VK_SHADER_STAGE_COMPUTE_BIT, "traversal/traversal_presort.comp.glsl", &options);

    res.compileShader(m_shaders.computeAssemblyVisibility, VK_SHADER_STAGE_COMPUTE_BIT, "traversal/assembly_visibility.comp.glsl", &options);
    if(config.useAssemblyCompact && config.useAssemblyPrefilter && !config.useSorting)
      res.compileShader(m_shaders.computeAssemblyCandidates,VK_SHADER_STAGE_COMPUTE_BIT,"traversal/assembly_candidates.comp.glsl",&options);

    res.compileShader(m_shaders.computeTraversalInit, VK_SHADER_STAGE_COMPUTE_BIT, "traversal/traversal_init.comp.glsl", &options);

    res.compileShader(m_shaders.computeTraversalRun, VK_SHADER_STAGE_COMPUTE_BIT, "traversal/traversal_run.comp.glsl", &options);

    res.compileShader(m_shaders.computeBuildSetup, VK_SHADER_STAGE_COMPUTE_BIT, "build/build_setup.comp.glsl", &options);

  if(config.useSeparateGroups)
  {

    res.compileShader(m_shaders.computeTraversalGroups, VK_SHADER_STAGE_COMPUTE_BIT,"traversal/traversal_run_separate_groups.comp.glsl", &options);
  }

  return res.verifyShaders(m_shaders);
}


bool RendererRasterClustersLod::init(Resources& res, RenderScene& rscene, const RendererConfig& config)
{
  m_resourceReservedUsage = {};
  m_config                = config;
  m_maxRenderClusters     = 1u << config.numRenderClusterBits;
  m_maxTraversalTasks     = 1u << config.numTraversalTaskBits;

  if(!initShaders(res, rscene, config))
  {
    return false;
  }


  m_pipelines = {};

  m_dsetPack.deinit();


  initBasics(res, rscene, config);


  m_resourceReservedUsage.geometryMemBytes   = rscene.getGeometrySize(true);
  m_resourceReservedUsage.operationsMemBytes = logMemoryUsage(rscene.getOperationsSize(), "operations", "rscene total");
  {

    res.createBuffer(m_sceneBuildBuffer, sizeof(shaderio::SceneBuilding), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);

    NVVK_DBG_NAME(m_sceneBuildBuffer.buffer);

    m_resourceReservedUsage.operationsMemBytes += logMemoryUsage(m_sceneBuildBuffer.bufferSize, "operations", "build shaderio");

    memset(&m_sceneBuildShaderio, 0, sizeof(m_sceneBuildShaderio));
    m_sceneBuildShaderio.numRenderInstances = uint32_t(m_renderInstances.size());
    m_sceneBuildShaderio.numAssemblyNodes   = uint32_t(rscene.scene->m_assemblyNodes.size());
    m_sceneBuildShaderio.assemblyCullingMinInstances = rscene.scene->m_config.assemblyCullingMinInstances;
    m_sceneBuildShaderio.assemblyLodPixelThreshold = rscene.scene->m_config.assemblyLodPixelThreshold;

    if(config.useAssemblyCompact && config.useAssemblyPrefilter && !config.useSorting
       && m_sceneBuildShaderio.numAssemblyNodes && m_sceneBuildShaderio.numRenderInstances)
    {
      const uint32_t invalid=~0u;
      std::vector<std::vector<uint32_t>> buckets(m_sceneBuildShaderio.numAssemblyNodes+1);
      for(uint32_t id=0;id<m_renderInstances.size();++id) {
        uint32_t a=m_renderInstances[id].assemblyID;
        buckets[a<m_sceneBuildShaderio.numAssemblyNodes?a:m_sceneBuildShaderio.numAssemblyNodes].push_back(id);
      }
      std::vector<shaderio::AssemblyCandidateRange> ranges;
      std::vector<uint32_t> members;members.reserve(m_renderInstances.size());
      constexpr uint32_t chunkSize=128;
      uint32_t activeAssemblies=0;
      for(uint32_t a=0;a<buckets.size();++a) {
        const auto& ids=buckets[a];
        if(!ids.empty() && a<m_sceneBuildShaderio.numAssemblyNodes) activeAssemblies++;
        uint32_t start=uint32_t(members.size());members.insert(members.end(),ids.begin(),ids.end());
        for(uint32_t first=0;first<ids.size();first+=chunkSize)
          ranges.push_back({a<m_sceneBuildShaderio.numAssemblyNodes?a:invalid,start+first,
             std::min(chunkSize,uint32_t(ids.size())-first),first==0?1u:0u});
      }
      res.createBuffer(m_candidateRanges,ranges.size()*sizeof(shaderio::AssemblyCandidateRange),VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
      res.createBuffer(m_candidateMembers,members.size()*sizeof(uint32_t),VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
      res.createBuffer(m_candidateInstances,members.size()*sizeof(uint32_t)*res.getFrameCycleCount(),VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
      res.simpleUploadBuffer(m_candidateRanges,ranges.data());res.simpleUploadBuffer(m_candidateMembers,members.data());
      m_sceneBuildShaderio.candidateRanges=m_candidateRanges.address;
      m_sceneBuildShaderio.candidateMembers=m_candidateMembers.address;
      m_sceneBuildShaderio.candidateInstances=m_candidateInstances.address;
      m_sceneBuildShaderio.numCandidateRanges=uint32_t(ranges.size());
      res.createBuffer(m_candidateControl,sizeof(shaderio::CandidateControl)*res.getFrameCycleCount(),VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);
      std::vector<shaderio::CandidateControl> control(res.getFrameCycleCount(),{0,{0,1,1}});res.simpleUploadBuffer(m_candidateControl,control.data());
      m_sceneBuildShaderio.candidateControl=m_candidateControl.address;
      const size_t extra=m_candidateRanges.bufferSize+m_candidateMembers.bufferSize+m_candidateInstances.bufferSize+m_candidateControl.bufferSize;
      m_resourceReservedUsage.operationsMemBytes+=extra;
      LOGI("COMPACT_ASSEMBLY_LAYOUT,%u,%u,%u,%zu\n",activeAssemblies,uint32_t(ranges.size()),uint32_t(members.size()),extra);
    }
    m_sceneBuildShaderio.maxRenderClusters = uint32_t(1u << config.numRenderClusterBits);

    m_sceneBuildShaderio.maxTraversalInfos = uint32_t(1u << config.numTraversalTaskBits);

    m_sceneBuildShaderio.indirectDispatchGroups.gridY  = 1;
    m_sceneBuildShaderio.indirectDispatchGroups.gridZ  = 1;
    m_sceneBuildShaderio.indirectDrawClustersEXT.gridZ = 1;
    m_sceneBuildShaderio.indirectDrawClustersNV.first  = 0;

    if(m_sceneBuildShaderio.numAssemblyNodes)
    {
      const size_t assemblyBytes = sizeof(shaderio::AssemblyNode) * rscene.scene->m_assemblyNodes.size();
      res.createBuffer(m_assemblyNodeBuffer, assemblyBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
      NVVK_DBG_NAME(m_assemblyNodeBuffer.buffer);
      res.simpleUploadBuffer(m_assemblyNodeBuffer, const_cast<shaderio::AssemblyNode*>(rscene.scene->m_assemblyNodes.data()));
      m_sceneBuildShaderio.assemblyNodes = m_assemblyNodeBuffer.address;
      m_resourceReservedUsage.operationsMemBytes += logMemoryUsage(m_assemblyNodeBuffer.bufferSize, "operations", "assembly nodes");

      const size_t assemblyStateBytes = sizeof(shaderio::AssemblyState) * rscene.scene->m_assemblyNodes.size();
      res.createBuffer(m_assemblyStateBuffer, assemblyStateBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
      NVVK_DBG_NAME(m_assemblyStateBuffer.buffer);
      m_sceneBuildShaderio.assemblyStates = m_assemblyStateBuffer.address;
      m_resourceReservedUsage.operationsMemBytes += logMemoryUsage(m_assemblyStateBuffer.bufferSize, "operations", "assembly states");
    }

    BufferRanges mem = {};

    m_sceneBuildShaderio.renderClusterInfos = mem.append(sizeof(shaderio::ClusterInfo) * m_sceneBuildShaderio.maxRenderClusters, 8);

    if(config.useSorting)
    {
      m_sceneBuildShaderio.instanceSortKeys   = mem.append(sizeof(uint32_t) * m_renderInstances.size(), 4);
      m_sceneBuildShaderio.instanceSortValues = mem.append(sizeof(uint32_t) * m_renderInstances.size(), 4);
    }

    if(config.useSeparateGroups)
    {
      m_sceneBuildShaderio.traversalGroupInfos = mem.append(sizeof(uint64_t) * m_sceneBuildShaderio.maxTraversalInfos, 8);
    }

    if(config.useTwoPassCulling && config.useCulling)
    {
      m_sceneBuildShaderio.instanceVisibility = mem.append(sizeof(uint8_t) * m_renderInstances.size(), 4);
    }

    res.createBuffer(m_sceneDataBuffer, mem.getSize(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    NVVK_DBG_NAME(m_sceneDataBuffer.buffer);

    m_resourceReservedUsage.operationsMemBytes += logMemoryUsage(m_sceneDataBuffer.bufferSize, "operations", "build data");

    m_sceneBuildShaderio.renderClusterInfos += m_sceneDataBuffer.address;
    m_sceneBuildShaderio.instanceSortKeys += m_sceneDataBuffer.address;
    m_sceneBuildShaderio.instanceSortValues += m_sceneDataBuffer.address;
    if(config.useSeparateGroups)
    {
      m_sceneBuildShaderio.traversalGroupInfos += m_sceneDataBuffer.address;
    }

    if(config.useTwoPassCulling && config.useCulling)
    {
      m_sceneBuildShaderio.instanceVisibility += m_sceneDataBuffer.address;
    }

    res.createBuffer(m_sceneTraversalBuffer, sizeof(uint64_t) * m_sceneBuildShaderio.maxTraversalInfos, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    NVVK_DBG_NAME(m_sceneTraversalBuffer.buffer);

    m_resourceReservedUsage.operationsMemBytes += logMemoryUsage(m_sceneTraversalBuffer.bufferSize, "operations", "build traversal");
    m_sceneBuildShaderio.traversalNodeInfos = m_sceneTraversalBuffer.address;
  }


  updateBasicDescriptors(res, rscene, &m_sceneBuildBuffer);

  if(rscene.useStreaming)
  {

    rscene.sceneStreaming.updateBindings(m_sceneBuildBuffer);
  }
  {

    m_stageFlags = VK_SHADER_STAGE_MESH_BIT_NV | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT;

    nvvk::DescriptorBindings bindings;

    bindings.addBinding(BINDINGS_FRAME_UBO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, m_stageFlags);

    bindings.addBinding(BINDINGS_READBACK_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, m_stageFlags);

    bindings.addBinding(BINDINGS_GEOMETRIES_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, m_stageFlags);

    bindings.addBinding(BINDINGS_RENDERINSTANCES_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, m_stageFlags);

    bindings.addBinding(BINDINGS_SCENEBUILDING_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, m_stageFlags);

    bindings.addBinding(BINDINGS_SCENEBUILDING_UBO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, m_stageFlags);


    bindings.addBinding(BINDINGS_HIZ_TEX, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, config.useCulling && config.useTwoPassCulling ? 2 : 1, m_stageFlags);

    if(rscene.useStreaming)
    {

      bindings.addBinding(BINDINGS_STREAMING_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, m_stageFlags);

      bindings.addBinding(BINDINGS_STREAMING_UBO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, m_stageFlags);
    }


    m_dsetPack.init(bindings, res.m_device);

    nvvk::createPipelineLayout(res.m_device, &m_pipelineLayout, {m_dsetPack.getLayout()}, {{m_stageFlags, 0, sizeof(uint32_t)}});

    nvvk::WriteSetContainer writeSets;
    writeSets.append(m_dsetPack.makeWrite(BINDINGS_FRAME_UBO), res.m_commonBuffers.frameConstants);
    writeSets.append(m_dsetPack.makeWrite(BINDINGS_READBACK_SSBO), &res.m_commonBuffers.readBack);
    writeSets.append(m_dsetPack.makeWrite(BINDINGS_GEOMETRIES_SSBO), rscene.getShaderGeometriesBuffer());
    writeSets.append(m_dsetPack.makeWrite(BINDINGS_RENDERINSTANCES_SSBO), m_renderInstanceBuffer);
    writeSets.append(m_dsetPack.makeWrite(BINDINGS_SCENEBUILDING_SSBO), m_sceneBuildBuffer);
    writeSets.append(m_dsetPack.makeWrite(BINDINGS_SCENEBUILDING_UBO), m_sceneBuildBuffer);


    writeSets.append(m_dsetPack.makeWrite(BINDINGS_HIZ_TEX, 0, 0), &res.m_hizUpdate[0].farImageInfo);
    if(config.useCulling && config.useTwoPassCulling)
    {
      writeSets.append(m_dsetPack.makeWrite(BINDINGS_HIZ_TEX, 0, 1), &res.m_hizUpdate[1].farImageInfo);
    }
    if(rscene.useStreaming)
    {
      writeSets.append(m_dsetPack.makeWrite(BINDINGS_STREAMING_SSBO), rscene.sceneStreaming.getShaderStreamingBuffer());
      writeSets.append(m_dsetPack.makeWrite(BINDINGS_STREAMING_UBO), rscene.sceneStreaming.getShaderStreamingBuffer());
    }

    vkUpdateDescriptorSets(res.m_device, uint32_t(writeSets.size()), writeSets.data(), 0, nullptr);
  }

  {
    nvvk::GraphicsPipelineCreator graphicsGen;
    nvvk::GraphicsPipelineState   state = res.m_basicGraphicsState;
    graphicsGen.pipelineInfo.layout                  = m_pipelineLayout;

    graphicsGen.renderingState.depthAttachmentFormat = res.m_frameBuffer.pipelineRenderingInfo.depthAttachmentFormat;
    graphicsGen.renderingState.stencilAttachmentFormat = res.m_frameBuffer.pipelineRenderingInfo.stencilAttachmentFormat;
    graphicsGen.colorFormats = {res.m_frameBuffer.colorFormat};
    state.rasterizationState.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    if (config.forceTwoSided)
    {
        state.rasterizationState.cullMode = VK_CULL_MODE_NONE;
    }
    graphicsGen.addShader(VK_SHADER_STAGE_MESH_BIT_NV, "main", nvvkglsl::GlslCompiler::getSpirvData(m_shaders.graphicsMesh));


    if(!m_config.useDepthOnly)
    {
      graphicsGen.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "main", nvvkglsl::GlslCompiler::getSpirvData(m_shaders.graphicsFragment));
    }


    graphicsGen.createGraphicsPipeline(res.m_device, nullptr, state, &m_pipelines.graphicsMesh);
  }

  {
    VkComputePipelineCreateInfo compInfo   = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    VkShaderModuleCreateInfo    shaderInfo = {};
    compInfo.stage                         = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    compInfo.stage.stage                   = VK_SHADER_STAGE_COMPUTE_BIT;
    compInfo.stage.pName                   = "main";
    compInfo.stage.pNext                   = &shaderInfo;
    compInfo.layout                        = m_pipelineLayout;

    if(config.useSorting)
    {

      shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.computeTraversalPresort);

      vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeTraversalPresort);
    }


    if(config.useAssemblyCompact && config.useAssemblyPrefilter && !config.useSorting) {
      shaderInfo=nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.computeAssemblyCandidates);
      NVVK_CHECK(vkCreateComputePipelines(res.m_device,nullptr,1,&compInfo,nullptr,&m_pipelines.computeAssemblyCandidates));
    }
    shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.computeAssemblyVisibility);

    vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeAssemblyVisibility);


    shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.computeBuildSetup);

    vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeBuildSetup);


    shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.computeTraversalInit);

    vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeTraversalInit);


    shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.computeTraversalRun);

    vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeTraversalRun);

    if(config.useSeparateGroups)
    {

      shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.computeTraversalGroups);

      vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeTraversalGroups);
    }
  }

  if(config.adaptiveTwoPassMode==2)
  {
    m_adaptiveCost.reset();
    m_lastCostValid=false;
    m_costSamples.resize(res.getFrameCycleCount());
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(res.m_physicalDevice,&properties);
    m_timestampPeriod=properties.limits.timestampPeriod;
    if(!m_costSamples.empty()&&m_timestampPeriod>0)
    {
      VkQueryPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
      poolInfo.queryType=VK_QUERY_TYPE_TIMESTAMP;poolInfo.queryCount=uint32_t(m_costSamples.size()*2);
      if(vkCreateQueryPool(res.m_device,&poolInfo,nullptr,&m_costQueries)!=VK_SUCCESS)
      { m_costQueries=VK_NULL_HANDLE;LOGW("Cost feedback unavailable; falling back to motion scheduler\n"); }
    }
  }

  return true;
}


static uint32_t getWorkGroupCount(uint32_t numThreads, uint32_t workGroupSize)
{
  return (numThreads + workGroupSize - 1) / workGroupSize;
}


void RendererRasterClustersLod::render(VkCommandBuffer cmd, Resources& res, RenderScene& rscene, const FrameConfig& frame, nvvk::ProfilerGpuTimer& profiler)
{
  VkMemoryBarrier memBarrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
  const AdaptiveCostSettings costSettings{m_config.twoPassCostProbeInterval,m_config.twoPassCostHysteresis,m_config.twoPassCostSmoothing,m_config.twoPassCostWarmupFrames};
  const uint32_t costSlot=res.getFrameCycleIndex();
  bool costSlotReady=m_costQueries!=VK_NULL_HANDLE&&costSlot<m_costSamples.size();
  if(costSlotReady&&m_costSamples[costSlot].pending)
  {
    uint64_t values[4]{};
    VkResult result=vkGetQueryPoolResults(res.m_device,m_costQueries,costSlot*2,2,sizeof(values),values,
        2*sizeof(uint64_t),VK_QUERY_RESULT_64_BIT|VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
    if(result==VK_SUCCESS&&values[1]&&values[3])
    {
      if(m_costSamples[costSlot].eligible)
        m_adaptiveCost.observe(m_costSamples[costSlot].dual,double(values[2]-values[0])*m_timestampPeriod/1e6,costSettings);
      m_costSamples[costSlot].pending=false;
    }
    else costSlotReady=false; // Never block the CPU waiting for a timing result.
  }



  {


    glm::vec2 renderScale = res.getFramebufferWindow2RenderScale();

    float     pixelScale  = std::min(renderScale.x, renderScale.y);


    m_sceneBuildShaderio.errorOverDistanceThreshold = clusterLodErrorOverDistance(frame.lodPixelError * pixelScale, frame.frameConstants.fov, frame.frameConstants.viewportf.y);
  }

  m_sceneBuildShaderio.traversalViewMatrix    = frame.traversalViewMatrix;
  m_sceneBuildShaderio.cullViewProjMatrix     = frame.cullViewProjMatrix;
  m_sceneBuildShaderio.cullViewProjMatrixLast = frame.cullViewProjMatrixLast;
  m_sceneBuildShaderio.pass                    = 0;
  m_sceneBuildShaderio.frameIndex              = m_frameIndex;

  const bool twoPassRequested = m_config.useCulling && m_config.useTwoPassCulling;
  const bool sceneLargeEnough = !rscene.scene || rscene.scene->m_totalClustersCount >= m_config.twoPassMinClusters;
  const bool cameraMovedEnough =
      matrixMaxAbsDelta(frame.cullViewProjMatrix, frame.cullViewProjMatrixLast) >= m_config.twoPassMatrixDelta;
  bool twoPassActive = twoPassRequested
                             && (!m_config.useAdaptiveTwoPassCulling
                                 || (!frame.freezeCulling && sceneLargeEnough && cameraMovedEnough));
  m_sceneBuildShaderio.adaptiveTemporalBypass=0;
  m_sceneBuildShaderio.adaptiveTriggerReason=0;
  m_sceneBuildShaderio.adaptiveMotionPixels=0;
  m_sceneBuildShaderio.adaptiveAccumulatedPixels=0;
  m_sceneBuildShaderio.adaptiveRefreshAge=0;
  float screenMotion=0;
  if(m_config.adaptiveTwoPassMode==1||m_config.adaptiveTwoPassMode==2)
  {
    const float motion=projectedSceneMotion(frame.cullViewProjMatrix,frame.cullViewProjMatrixLast,
        rscene.scene->m_gridBbox.lo,rscene.scene->m_gridBbox.hi,frame.frameConstants.viewportf);
    screenMotion=motion;
    AdaptiveTwoPassSettings settings{m_config.twoPassMotionPixels,m_config.twoPassFastPixels,m_config.twoPassMaxInterval};
    const auto decision=(m_config.adaptiveTwoPassMode==2&&m_costQueries!=VK_NULL_HANDLE)
        ? m_adaptiveCost.update(motion,twoPassRequested,m_config.useAdaptiveTwoPassCulling,
            sceneLargeEnough,frame.freezeCulling,rscene.useStreaming,costSettings)
        : m_adaptiveTwoPass.update(motion,twoPassRequested,m_config.useAdaptiveTwoPassCulling,
            sceneLargeEnough,frame.freezeCulling,rscene.useStreaming,settings);
    twoPassActive=decision.active;
    m_sceneBuildShaderio.adaptiveTemporalBypass=decision.bypassTemporal ? 1u : 0u;
    m_sceneBuildShaderio.adaptiveTriggerReason=decision.reason;
    m_sceneBuildShaderio.adaptiveMotionPixels=motion;
    m_sceneBuildShaderio.adaptiveAccumulatedPixels=decision.accumulatedPixels;
    m_sceneBuildShaderio.adaptiveRefreshAge=decision.age;
  }
  // Offline quality replay uses the decisions recorded in its paired timing
  // run, so image transfer/compression cannot change the adaptive schedule.
  if(res.benchmarkForcedTwoPass >= 0)
  {
    twoPassActive=res.benchmarkForcedTwoPass!=0;
    m_sceneBuildShaderio.adaptiveTemporalBypass=res.benchmarkForcedBypass;
    m_sceneBuildShaderio.adaptiveTriggerReason=res.benchmarkForcedReason;
  }
  if(m_sceneBuildShaderio.adaptiveTemporalBypass)
    m_sceneBuildShaderio.cullViewProjMatrixLast=frame.cullViewProjMatrix;
  m_sceneBuildShaderio.twoPassCullingActive = twoPassActive ? 1u : 0u;
  // The diagnostic reference keeps current-frustum and size/LOD rules while
  // disabling every Hi-Z lookup through USE_HIZ_OCCLUSION.
  if(!m_config.useHizOcclusion)
    m_sceneBuildShaderio.cullViewProjMatrixLast = frame.cullViewProjMatrix;
  if(m_config.adaptiveTwoPassMode==2&&m_config.useRenderStats&&twoPassRequested
      &&m_config.useAdaptiveTwoPassCulling&&m_frameIndex%60==0)
    LOGI("COST_FEEDBACK_CSV,%u,%.6f,%.6f,%u,%u\n",m_frameIndex,m_adaptiveCost.singleCost(),
        m_adaptiveCost.dualCost(),twoPassActive?1u:0u,m_sceneBuildShaderio.adaptiveTriggerReason);
  const bool measureCost=costSlotReady&&twoPassRequested&&m_config.useAdaptiveTwoPassCulling
      &&sceneLargeEnough&&!frame.freezeCulling&&!rscene.useStreaming
      &&(screenMotion>.001f||(m_sceneBuildShaderio.adaptiveTriggerReason&(AdaptiveCostController::CostDual|AdaptiveCostController::CostSingle))!=0);
  if(measureCost)
  {
    vkCmdResetQueryPool(cmd,m_costQueries,costSlot*2,2);
    vkCmdWriteTimestamp(cmd,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,m_costQueries,costSlot*2);
  }



  if(m_sceneBuildShaderio.numCandidateRanges) {
    m_sceneBuildShaderio.candidateInstances=m_candidateInstances.address+VkDeviceSize(res.getFrameCycleIndex())*m_sceneBuildShaderio.numRenderInstances*sizeof(uint32_t);
    m_sceneBuildShaderio.candidateControl=m_candidateControl.address+VkDeviceSize(res.getFrameCycleIndex())*sizeof(shaderio::CandidateControl);
  }
  vkCmdUpdateBuffer(cmd, res.m_commonBuffers.frameConstants.buffer, 0, sizeof(shaderio::FrameConstants), (const uint32_t*)&frame.frameConstants);
  vkCmdUpdateBuffer(cmd, m_sceneBuildBuffer.buffer, 0, sizeof(shaderio::SceneBuilding), (const uint32_t*)&m_sceneBuildShaderio);

  vkCmdFillBuffer(cmd, res.m_commonBuffers.readBack.buffer, 0, sizeof(shaderio::Readback), 0);
  if(m_sceneBuildShaderio.numCandidateRanges) vkCmdFillBuffer(cmd,m_candidateControl.buffer,VkDeviceSize(res.getFrameCycleIndex())*sizeof(shaderio::CandidateControl),2*sizeof(uint32_t),0);


  vkCmdFillBuffer(cmd, m_sceneTraversalBuffer.buffer, 0, m_sceneTraversalBuffer.bufferSize, ~0);

  if(rscene.useStreaming)
  {
    SceneStreaming::FrameSettings settings;
    settings.ageThreshold = frame.streamingAgeThreshold;

    rscene.sceneStreaming.cmdBeginFrame(cmd, res.m_queueStates.primary, res.m_queueStates.transfer, settings, profiler);
  }


  memBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       0, 1, &memBarrier, 0, nullptr, 0, nullptr);

  if(rscene.useStreaming)
  {

    rscene.sceneStreaming.cmdPreTraversal(cmd, profiler);


    memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_DEPENDENCY_BY_REGION_BIT, 1, &memBarrier, 0, nullptr, 0, nullptr);
  }

  const uint32_t passCount = twoPassActive ? 2 : 1;
  const uint32_t lastPass  = passCount - 1;
  for(uint32_t pass = 0; pass < passCount; pass++)
  {
    if(pass == 1) res.benchmarkTimestamp(cmd, 8);
    res.benchmarkTimestamp(cmd, pass == 0 ? 2 : 4);
    {

      auto timerSection = profiler.cmdFrameSection(cmd, "Traversal Preparation");

      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, m_dsetPack.getSetPtr(), 0, nullptr);

      if(pass == 0 && m_config.useSorting)
      {

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeTraversalPresort);
        res.cmdLinearDispatch(cmd, getWorkGroupCount(m_sceneBuildShaderio.numRenderInstances, TRAVERSAL_PRESORT_WORKGROUP));


        memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_DEPENDENCY_BY_REGION_BIT, 1, &memBarrier, 0, nullptr, 0, nullptr);

        vrdxCmdSortKeyValue(cmd, res.m_vrdxSorter, m_sceneBuildShaderio.numRenderInstances, m_sceneDataBuffer.buffer,
                            m_sceneBuildShaderio.instanceSortKeys - m_sceneDataBuffer.address, m_sceneDataBuffer.buffer,
                            m_sceneBuildShaderio.instanceSortValues - m_sceneDataBuffer.address,
                            m_sortingAuxBuffer.buffer, 0, nullptr, 0);


        memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_DEPENDENCY_BY_REGION_BIT, 1, &memBarrier, 0, nullptr, 0, nullptr);
      }

      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, m_dsetPack.getSetPtr(), 0, nullptr);

      res.benchmarkTimestamp(cmd, pass==0?12:16);
      if(m_sceneBuildShaderio.numCandidateRanges)
      {
        vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,m_pipelines.computeAssemblyCandidates);
        res.cmdLinearDispatch(cmd,m_sceneBuildShaderio.numCandidateRanges);
        memBarrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;
        memBarrier.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
        vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                             0,1,&memBarrier,0,nullptr,0,nullptr);
      }
      else if(m_config.useAssemblyPrefilter && m_sceneBuildShaderio.numAssemblyNodes)
      {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeAssemblyVisibility);
        res.cmdLinearDispatch(cmd, getWorkGroupCount(m_sceneBuildShaderio.numAssemblyNodes, ASSEMBLY_VISIBILITY_WORKGROUP));

        memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_DEPENDENCY_BY_REGION_BIT, 1, &memBarrier, 0, nullptr, 0, nullptr);
      }

      res.benchmarkTimestamp(cmd,pass==0?13:17);
      res.benchmarkTimestamp(cmd,pass==0?14:18);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeTraversalInit);
      if(m_sceneBuildShaderio.numCandidateRanges)
        vkCmdDispatchIndirect(cmd,m_candidateControl.buffer,VkDeviceSize(res.getFrameCycleIndex())*sizeof(shaderio::CandidateControl)+offsetof(shaderio::CandidateControl,dispatch));
      else
        res.cmdLinearDispatch(cmd, getWorkGroupCount(m_sceneBuildShaderio.numRenderInstances, TRAVERSAL_INIT_WORKGROUP));


      memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_DEPENDENCY_BY_REGION_BIT, 1, &memBarrier, 0, nullptr, 0, nullptr);

      res.benchmarkTimestamp(cmd,pass==0?15:19);
      uint32_t buildSetupID = BUILD_SETUP_TRAVERSAL_RUN;

      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeBuildSetup);
      vkCmdPushConstants(cmd, m_pipelineLayout, m_stageFlags, 0, sizeof(uint32_t), &buildSetupID);

      vkCmdDispatch(cmd, 1, 1, 1);


      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_DEPENDENCY_BY_REGION_BIT, 1, &memBarrier, 0, nullptr, 0, nullptr);
    }
    {

        auto timerSection = profiler.cmdFrameSection(cmd, "Traversal Run");


        res.benchmarkTimestamp(cmd,pass==0?20:22);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeTraversalRun);
        res.cmdLinearDispatch(cmd, getWorkGroupCount(frame.traversalPersistentThreads, TRAVERSAL_RUN_WORKGROUP));

      if(m_config.useSeparateGroups)
      {
        memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;

        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 1, &memBarrier, 0, nullptr, 0, nullptr);

        {
          auto groupTimer = profiler.cmdFrameSection(cmd, "Group Visibility");
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeTraversalGroups);
        vkCmdDispatchIndirect(cmd, m_sceneBuildBuffer.buffer, offsetof(shaderio::SceneBuilding, indirectDispatchGroups));
        }
      }


      memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_DEPENDENCY_BY_REGION_BIT, 1, &memBarrier, 0, nullptr, 0, nullptr);

      res.benchmarkTimestamp(cmd,pass==0?21:23);
      uint32_t buildSetupID = BUILD_SETUP_DRAW;

      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeBuildSetup);
      vkCmdPushConstants(cmd, m_pipelineLayout, m_stageFlags, 0, sizeof(uint32_t), &buildSetupID);

      vkCmdDispatch(cmd, 1, 1, 1);


      memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_MESH_SHADER_BIT_NV | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                           VK_DEPENDENCY_BY_REGION_BIT, 1, &memBarrier, 0, nullptr, 0, nullptr);
    }

    res.benchmarkTimestamp(cmd, pass == 0 ? 3 : 5);
    if(pass == lastPass && rscene.useStreaming)
    {

      rscene.sceneStreaming.cmdPostTraversal(cmd, true, profiler);
    }
    else if(rscene.useStreaming)
    {

      auto timerSection = profiler.cmdFrameSection(cmd, "Stream Post Traversal");
    }
    {

       auto timerSection = profiler.cmdFrameSection(cmd, "Draw");

      VkAttachmentLoadOp op = pass == 1 ? VK_ATTACHMENT_LOAD_OP_LOAD : (m_config.useShading ? VK_ATTACHMENT_LOAD_OP_DONT_CARE : VK_ATTACHMENT_LOAD_OP_CLEAR);

      res.cmdBeginRendering(cmd, false, op, op);
      if(pass == 0 && m_config.useShading)
      {

        writeBackgroundSky(cmd);
      }

      {

        auto timerSection = profiler.cmdFrameSection(cmd, "HW-Raster");

        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, m_dsetPack.getSetPtr(), 0, nullptr);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelines.graphicsMesh);

        if(m_config.useEXTmeshShader)
        {
          vkCmdDrawMeshTasksIndirectEXT(cmd, m_sceneBuildBuffer.buffer, offsetof(shaderio::SceneBuilding, indirectDrawClustersEXT), 1, 0);
        }
        else
        {
          vkCmdDrawMeshTasksIndirectNV(cmd, m_sceneBuildBuffer.buffer, offsetof(shaderio::SceneBuilding, indirectDrawClustersNV), 1, 0);
        }
      }

      if(frame.showClusterBboxes)
      {

        renderClusterBboxes(cmd, m_sceneBuildBuffer);
      }

      if(pass == lastPass && frame.showInstanceBboxes)
      {

        renderInstanceBboxes(cmd);
      }

      vkCmdEndRendering(cmd);
    }


    if(pass == 1) res.benchmarkTimestamp(cmd, 9);
    if(pass == 0 && twoPassActive) res.benchmarkTimestamp(cmd, 6);
    if(!frame.freezeCulling)
    {
      if(twoPassActive)
      {

        res.cmdBuildHiz(cmd, frame, profiler, pass ^ 1);
      }
      else
      {

        res.cmdBuildHiz(cmd, frame, profiler, 0);
      }
    }
    if(pass == 0 && twoPassActive) res.benchmarkTimestamp(cmd, 7);
  }

  if(passCount > 1)
  {

    profiler.getProfilerTimeline()->frameAccumulationSplit();
  }


  if(rscene.useStreaming)
  {

    rscene.sceneStreaming.cmdEndFrame(cmd, res.m_queueStates.primary, profiler);
  }


  m_resourceReservedUsage.geometryMemBytes = rscene.getGeometrySize(true);
  m_resourceActualUsage                    = m_resourceReservedUsage;

  m_resourceActualUsage.geometryMemBytes   = rscene.getGeometrySize(false);
  if(measureCost)
  {
    vkCmdWriteTimestamp(cmd,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,m_costQueries,costSlot*2+1);
    const bool eligible=m_lastCostValid&&m_lastCostFrame+1==m_frameIndex&&m_lastCostDual==twoPassActive;
    m_costSamples[costSlot]={true,twoPassActive,eligible};
    m_lastCostValid=true;m_lastCostFrame=m_frameIndex;m_lastCostDual=twoPassActive;
  }
  m_frameIndex++;
}


void RendererRasterClustersLod::updatedFrameBuffer(Resources& res, RenderScene& rscene)
{
  m_adaptiveTwoPass.reset();

  vkDeviceWaitIdle(res.m_device);
  m_adaptiveCost.reset();
  m_lastCostValid=false;
  for(auto& sample:m_costSamples)sample.pending=false;


  VkWriteDescriptorSet writes[2];

  writes[0] = m_dsetPack.makeWrite(BINDINGS_HIZ_TEX, 0, 0);
  writes[0].pImageInfo = &res.m_hizUpdate[0].farImageInfo;

  if(m_config.useCulling && m_config.useTwoPassCulling)
  {

    writes[1]            = m_dsetPack.makeWrite(BINDINGS_HIZ_TEX, 0, 1);
    writes[1].pImageInfo = &res.m_hizUpdate[1].farImageInfo;
  }


  vkUpdateDescriptorSets(res.m_device, m_config.useCulling && m_config.useTwoPassCulling ? 2 : 1, writes, 0, nullptr);

  Renderer::updatedFrameBuffer(res, rscene);
}


void RendererRasterClustersLod::deinit(Resources& res)
{
  if(m_costQueries)vkDestroyQueryPool(res.m_device,m_costQueries,nullptr);
  m_costQueries=VK_NULL_HANDLE;m_costSamples.clear();m_adaptiveCost.reset();



  res.destroyPipelines(m_pipelines);


  vkDestroyPipelineLayout(res.m_device, m_pipelineLayout, nullptr);


  m_dsetPack.deinit();


  res.m_allocator.destroyBuffer(m_sceneDataBuffer);

  res.m_allocator.destroyBuffer(m_assemblyNodeBuffer);

  res.m_allocator.destroyBuffer(m_assemblyStateBuffer);
  res.m_allocator.destroyBuffer(m_candidateRanges);res.m_allocator.destroyBuffer(m_candidateMembers);res.m_allocator.destroyBuffer(m_candidateInstances);res.m_allocator.destroyBuffer(m_candidateControl);

  res.m_allocator.destroyBuffer(m_sceneBuildBuffer);

  res.m_allocator.destroyBuffer(m_sceneTraversalBuffer);

  deinitBasics(res);
}


std::unique_ptr<Renderer> makeRendererRasterClustersLod()
{
  return std::make_unique<RendererRasterClustersLod>();
}
}
