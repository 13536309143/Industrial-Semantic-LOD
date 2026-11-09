//==============================================================================
// src/streaming/streaming.hpp
// Declares the high-level scene streaming controller.
// The controller owns persistent per-geometry metadata, shader buffers, streaming compute pipelines, and per-frame task orchestration.
//==============================================================================



#pragma once


#include "streamutils.hpp"


namespace lodclusters {


class SceneStreaming
{
public:


  bool init(Resources* res, const Scene* scene, const StreamingConfig& config);


  void deinit();


  void reset();


  bool reloadShaders()
  {

    deinitShadersAndPipelines();
    return initShadersAndPipelines();
  }


  struct FrameSettings
  {
    uint32_t ageThreshold = 16;
  };
  void cmdBeginFrame(VkCommandBuffer         cmd,
                     QueueState&             cmdQueueState,
                     QueueState&             asyncQueueState,
                     const FrameSettings&    settings,
                     nvvk::ProfilerGpuTimer& profiler);


  void cmdPreTraversal(VkCommandBuffer cmd, nvvk::ProfilerGpuTimer& profiler);


  void cmdPostTraversal(VkCommandBuffer cmd, bool runAgeFilter, nvvk::ProfilerGpuTimer& profiler);


  void cmdEndFrame(VkCommandBuffer cmd, QueueState& cmdQueueState, nvvk::ProfilerGpuTimer& profiler);


  void getStats(StreamingStats& stats) const;
  const nvvk::BufferTyped<shaderio::Geometry>& getShaderGeometriesBuffer() const { return m_shaderGeometriesBuffer; }
  const nvvk::Buffer&                          getShaderStreamingBuffer() const { return m_shaderBuffer; }
  const shaderio::SceneStreaming&              getShaderStreamingData() const { return m_shaderData; }
  const StreamingConfig&                       getStreamingConfig() const { return m_config; }


  size_t getGeometrySize(bool reserved) const;
  size_t getOperationsSize() const { return m_operationsSize; }


  void updateBindings(const nvvk::Buffer& sceneBuildingBuffer);
#ifndef NDEBUG
  static const int32_t s_defaultDebugFrameLimit = -1;
#else
  static const int32_t s_defaultDebugFrameLimit = -1;
#endif
  int32_t m_debugFrameLimit = s_defaultDebugFrameLimit;

private:
  Resources*   m_resources = nullptr;
  const Scene* m_scene     = nullptr;

  StreamingConfig m_config;
  size_t          m_persistentGeometrySize;
  size_t          m_operationsSize;
  size_t          m_peakGeometrySize = 0;
  uint32_t        m_peakFrameIndex   = ~0;
  uint32_t        m_lastUpdateIndex;
  uint32_t        m_frameIndex;
  StreamingStats  m_stats;


  struct PersistentGeometry
  {
    nvvk::BufferTyped<shaderio::Node>     nodes;
    nvvk::BufferTyped<shaderio::BBox>     nodeBboxes;
    nvvk::BufferTyped<shaderio::LodLevel> lodLevels;
    nvvk::BufferTyped<uint64_t>           groupAddresses;
    std::vector<nvvk::Buffer>             lowDetailGroupsData;
    uint32_t                              lodLevelsCount                                = 0;
    uint32_t                              lodLoadedGroupsCount[SHADERIO_MAX_LOD_LEVELS] = {};
    uint32_t                              lodGroupsCount[SHADERIO_MAX_LOD_LEVELS]       = {};
  };

  std::vector<PersistentGeometry>       m_persistentGeometries;
  std::vector<shaderio::Geometry>       m_shaderGeometries;
  nvvk::BufferTyped<shaderio::Geometry> m_shaderGeometriesBuffer;


  void initGeometries(Resources& res, const Scene* scene);


  void resetGeometryGroupAddresses(Resources::BatchedUploader& uploader);
  shaderio::SceneStreaming m_shaderData;
  nvvk::Buffer             m_shaderBuffer;
  StreamingTaskQueue m_requestsTaskQueue;
  StreamingTaskQueue m_storageTaskQueue;
  StreamingTaskQueue m_updatesTaskQueue;
  StreamingRequests m_requests;
  StreamingResident m_resident;
  StreamingStorage m_storage;
  StreamingUpdates m_updates;
  uint32_t handleCompletedRequest(VkCommandBuffer      cmd,
                                  QueueState&          cmdQueueState,
                                  QueueState&          asyncQueueState,
                                  const FrameSettings& settings,
                                  uint32_t             popRequestIndex);


  struct Shaders
  {
    shaderc::SpvCompilationResult computeAgeFilterGroups;
    shaderc::SpvCompilationResult computeUpdateSceneRaster;
  };


  struct Pipelines
  {
    VkPipeline computeAgeFilterGroups   = nullptr;
    VkPipeline computeUpdateSceneRaster = nullptr;
  };

  Shaders              m_shaders;
  Pipelines            m_pipelines;
  VkPipelineLayout     m_pipelineLayout{};
  nvvk::DescriptorPack m_dsetPack;


  bool initShadersAndPipelines();


  void deinitShadersAndPipelines();
};
}
