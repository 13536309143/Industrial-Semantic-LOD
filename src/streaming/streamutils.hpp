//==============================================================================
// src/streaming/streamutils.hpp
// Declares CPU-side streaming queues, residency tables, transfer staging, and allocator helpers.
// These helpers translate GPU load/unload requests into bounded upload tasks and stable resident group IDs.
//==============================================================================



#pragma once


#include <queue>
#include <nvutils/logger.hpp>
#include <nvutils/id_pool.hpp>
#include <nvvk/buffer_suballocator.hpp>
#include <nvvk/command_pools.hpp>
#include "scene.hpp"
#include "resources.hpp"
#include "shaderio_streaming.h"


namespace lodclusters {

static const uint32_t STREAMING_MAX_ACTIVE_TASKS = 3;
static const uint32_t INVALID_TASK_INDEX         = ~0;


struct StreamingConfig
{
  bool useAsyncTransfer           = false;
  bool useDecoupledAsyncTransfer  = false;

  uint32_t maxPerFrameLoadRequests   = 128;
  uint32_t maxPerFrameUnloadRequests = 1024;

  uint32_t maxGroups   = 1 << 16;
  uint32_t maxClusters = 0;

  size_t maxTransferMegaBytes    = 32;
  size_t maxGeometryMegaBytes    = 1024 * 2;
};


struct StreamingStats
{
  uint32_t residentGroups   = 0;
  uint32_t residentClusters = 0;
  uint32_t maxGroups        = 0;
  uint32_t maxClusters      = 0;

  uint32_t persistentGroups    = 0;
  uint32_t persistentClusters  = 0;
  uint64_t persistentDataBytes = 0;

  uint64_t maxDataBytes      = 0;
  uint64_t reservedDataBytes = 0;
  uint64_t usedDataBytes     = 0;

  uint32_t maxSizedLeft      = 0;
  uint32_t maxSizedReserved  = 0;

  uint64_t maxTransferBytes     = 0;
  uint64_t transferBytes        = 0;
  uint64_t geometryUploadBytes=0, requestedDeviceBytes=0;
  uint32_t transferCount        = 0;
  uint32_t loadCount            = 0;
  uint32_t unloadCount          = 0;
  uint32_t uncompletedLoadCount = 0;
  uint32_t maxLoadCount         = 0;
  uint32_t maxUnloadCount       = 0;

  uint32_t couldNotAllocateGroup = 0;
  uint32_t couldNotTransfer      = 0;
  uint32_t couldNotStore         = 0;
};

union GeometryGroup
{
  struct
  {
    uint32_t geometryID;
    uint32_t groupID;
  };
  uint64_t key;
};


// CPU mirror of GPU load/unload request buffers.
class StreamingRequests
{
public:


  struct TaskInfo
  {
    const shaderio::StreamingRequest* shaderData;
    const GeometryGroup*              loadGeometryGroups;
    const GeometryGroup*              unloadGeometryGroups;
  };


  void init(Resources& res, const StreamingConfig& config, uint32_t groupCountAlignment, uint32_t clusterCountAlignment);


  void deinit(Resources& res);


  size_t getOperationsSize() const;


  void applyTask(shaderio::StreamingRequest& shaderData, uint32_t taskIndex, uint32_t frameIndex);


  void cmdRunTask(VkCommandBuffer cmd, const shaderio::StreamingRequest& shaderData, VkBuffer srcBuffer, size_t srcBufferOffset);


  const TaskInfo& getCompletedTask(uint32_t taskIndex) { return m_taskInfos[taskIndex]; }

private:
  nvvk::Buffer m_requestBuffer;
  nvvk::Buffer m_requestHostBuffer;

  uint64_t m_requestSize;
  uint64_t m_shaderDataOffset;

  shaderio::StreamingRequest m_shaderData;
  TaskInfo                   m_taskInfos[STREAMING_MAX_ACTIVE_TASKS];
};


// Tracks which geometry groups currently own resident IDs and GPU memory addresses.
class StreamingResident
{
public:
  static const uint32_t INVALID_GROUP = ~0;


  struct Group
  {
    GeometryGroup             geometryGroup;
    uint32_t                  activeIndex;
    uint32_t                  groupResidentID;
    uint32_t                  clusterResidentID;
    uint16_t                  clusterCount;
    uint16_t                  lodLevel;
    uint64_t                  deviceAddress;
    nvvk::BufferSubAllocation storageHandle;
  };


  void init(Resources& res, const StreamingConfig& config, uint32_t groupCountAlignment, uint32_t clusterCountAlignment);


  void         deinit(Resources& res);


  void         reset(shaderio::StreamingResident& shaderData);


  size_t getOperationsSize() const;


  void   getStats(StreamingStats& stats) const;


  void                uploadInitialState(Resources::BatchedUploader& uploader, shaderio::StreamingResident& shaderData);


  const StreamingResident::Group* findGroup(GeometryGroup geometryGroup) const;
  const StreamingResident::Group& getGroup(uint32_t groupResidentID) const { return m_groups[groupResidentID]; }


  uint32_t getLoadActiveGroupsOffset() const;


  uint32_t getLoadActiveClustersOffset() const;


  bool                      canAllocateGroup(uint32_t numClusters) const;


  StreamingResident::Group* addGroup(GeometryGroup geometryGroup, uint32_t clusterCount);


  void                      removeGroup(uint32_t groupResidentID);


  size_t cmdUploadTask(VkCommandBuffer cmd, uint32_t taskIndex);


  void applyTask(shaderio::StreamingResident& shaderData, uint32_t taskIndex, uint32_t frameIndex);


  void cmdRunTask(VkCommandBuffer cmd, uint32_t taskIndex);


private:


  struct UpdateRange
  {

    uint32_t lo = uint32_t(~0);
    uint32_t hi = 0;


    void update(uint32_t index)
    {

      lo = std::min(lo, index);

      hi = std::max(hi, index);
    }

    uint32_t count() const { return hi == 0 && lo == ~0 ? 0 : 1 + hi - lo; }
  };


  struct TaskInfo
  {
    VkBufferCopy                region;
    shaderio::StreamingResident shaderData;
  };


  std::unordered_map<uint64_t, uint32_t> m_mapGeometryGroup2Residency;

  nvutils::IDPool m_groupAllocator;
  nvutils::IDPool m_clusterAllocator;

  uint32_t m_maxClusters;
  uint32_t m_maxGroups;

  std::vector<Group> m_groups;


  std::vector<uint32_t> m_activeGroupIndices;

  uint32_t m_lowDetailGroupsCount;
  uint32_t m_lowDetailClustersCount;
  uint32_t m_lowDetailMaxGroupClusters;

  uint32_t m_activeGroupsCount;
  uint32_t m_activeClustersCount;

  nvvk::Buffer m_residentBuffer;
  uint64_t     m_residentGroupsOffset;
  uint64_t     m_residentGroupIDsOffset;
  uint64_t     m_residentClustersOffset;
  uint64_t     m_residentActiveOffset;
  uint64_t     m_residentActiveUpdateOffset;

  shaderio::StreamingResident m_shaderData;

  nvvk::BufferTyped<uint32_t> m_residentActiveHostBuffer;
  UpdateRange                 m_groupIndicesUpdateRange;

  TaskInfo m_taskInfos[STREAMING_MAX_ACTIVE_TASKS];
};


// Accumulates address patches that publish finished uploads back to shader-visible scene data.
class StreamingUpdates
{
public:


  struct TaskInfo
  {
    uint32_t                          loadCount;
    uint32_t                          unloadCount;
    uint32_t                          newClusterCount;
    uint32_t                          loadActiveGroupsOffset;
    uint32_t                          loadActiveClustersOffset;
    shaderio::StreamingPatch*         loadPatches;
    shaderio::StreamingPatch*         unloadPatches;
    nvvk::BufferSubAllocation*        unloadHandles;
  };


  struct NewInfo
  {
    uint32_t groups   = 0;
    uint32_t clusters = 0;
  };


  void init(Resources& res, const StreamingConfig& config, uint32_t groupCountAlignment, uint32_t clusterCountAlignment);


  void deinit(Resources& res);


  size_t   getOperationsSize() const;


  void reset();


  NewInfo getFutureNew(uint64_t frameIndex) const
  {


    NewInfo info = m_pendingNew;


    for(uint32_t i = 0; i < STREAMING_MAX_ACTIVE_TASKS; i++)
    {
      if(m_scheduledNewFrame[i] > frameIndex)
      {
        info.groups += m_scheduledNew[i].groups;
        info.clusters += m_scheduledNew[i].clusters;
      }
    }
    return info;
  }


  TaskInfo& getNewTask(uint32_t taskIndex);


  size_t cmdUploadTask(VkCommandBuffer cmd, uint32_t taskIndex);


  void applyTask(shaderio::StreamingUpdate& shaderData, uint32_t taskIndex, uint32_t frameIndex);


  const TaskInfo& getCompletedTask(uint32_t taskIndex) const { return m_taskInfos[taskIndex]; }

private:
  nvvk::BufferTyped<shaderio::StreamingPatch> m_patchesBuffer;
  nvvk::BufferTyped<shaderio::StreamingPatch> m_patchesHostBuffer;

  std::vector<nvvk::BufferSubAllocation> m_unloadHandles;
  TaskInfo                               m_taskInfos[STREAMING_MAX_ACTIVE_TASKS];

  shaderio::StreamingUpdate m_shaderData;

  uint32_t m_clusterCountAlignment;
  uint32_t m_scheduleIndex;
  NewInfo  m_pendingNew;
  NewInfo  m_scheduledNew[STREAMING_MAX_ACTIVE_TASKS]      = {};
  uint64_t m_scheduledNewFrame[STREAMING_MAX_ACTIVE_TASKS] = {};
};


// Owns the bounded GPU memory pool and staging space used by streaming transfers.
class StreamingStorage
{
public:


  struct TaskInfo
  {
    size_t usedMemory;
    size_t baseOffset;
    size_t regionCount;
  };


  void init(Resources& res, const StreamingConfig& config);


  void deinit(Resources& res);


  void reset();


  void free(nvvk::BufferSubAllocation& handle);


  void   getStats(StreamingStats& stats) const;


  size_t getOperationsSize() const;


  size_t getMaxDataSize() const;


  TaskInfo& getNewTask(uint32_t taskIndex);


  bool canTransfer(const TaskInfo& operation, size_t size) const;


  bool allocate(nvvk::BufferSubAllocation& handle, GeometryGroup group, size_t sz, uint64_t& deviceAddress);


  void* appendTransfer(TaskInfo& operation, const nvvk::BufferSubAllocation& dstHandle);


  uint32_t cmdUploadTask(VkCommandBuffer cmd);

  nvvk::ManagedCommandPools m_taskCommandPool;

private:


  struct CopyInfo
  {
    VkBuffer targetBuffer;
    size_t   regionOffset;
    size_t   regionCount;
  };

  size_t m_maxSceneBytes;
  size_t m_maxTransferBytes;
  size_t m_blockBytes;

  nvvk::Buffer                       m_transferHostBuffer;
  nvvk::BufferSubAllocator::InitInfo m_dataInfo;
  nvvk::BufferSubAllocator           m_dataAllocator;
  std::vector<uint32_t>              m_dataQueueFamilies;

  std::vector<CopyInfo>     m_copyInfos;
  std::vector<VkBufferCopy> m_copyRegions;

  TaskInfo m_taskOperations[STREAMING_MAX_ACTIVE_TASKS];
};


// Small fixed-depth task queue used to overlap CPU scheduling with GPU transfer completion.
class StreamingTaskQueue
{
public:

  static_assert(STREAMING_MAX_ACTIVE_TASKS < 32);

  StreamingTaskQueue() { m_availableTaskBits = (1 << STREAMING_MAX_ACTIVE_TASKS) - 1; }


  uint32_t acquireTaskIndex()
  {

    for(uint32_t i = 0; i < STREAMING_MAX_ACTIVE_TASKS; i++)
    {
      if(m_availableTaskBits & (1 << i))
      {

        m_availableTaskBits &= ~(1 << i);
        return i;
      }
    }

    return INVALID_TASK_INDEX;
  }


  void releaseTaskIndex(uint32_t index)
  {
    assert((m_availableTaskBits & (1 << index)) == 0);
    m_availableTaskBits |= (1 << index);
  }


  bool canPop(VkDevice device, bool ensureAcquisition)
  {
    if(ensureAcquisition && !m_availableTaskBits && !m_taskQueue.empty())
    {


      if(m_taskQueue.front().semaphoreState.wait(device, ~0ULL) == VK_TIMEOUT)
      {

        LOGE("Failure to wait for semaphore");
        {

          exit(-1);
        }
      }
    }

    return !m_taskQueue.empty() && m_taskQueue.front().semaphoreState.testSignaled(device);
  }


  void push(uint32_t taskIndex, nvvk::SemaphoreState semaphoreState, uint32_t dependentIndex = INVALID_TASK_INDEX)
  {
    Task task = {
        .semaphoreState = semaphoreState,
        .taskIndex      = taskIndex,
        .dependentIndex = dependentIndex,
    };

    m_taskQueue.push(task);
  }


  uint32_t pop()
  {
    uint32_t taskIndex = m_taskQueue.front().taskIndex;

    assert(taskIndex != INVALID_TASK_INDEX);

    m_taskQueue.pop();
    return taskIndex;
  }


  uint32_t popWithDependent(uint32_t& dependentIndex)
  {
    uint32_t taskIndex = m_taskQueue.front().taskIndex;

    assert(taskIndex != INVALID_TASK_INDEX);
    dependentIndex = m_taskQueue.front().dependentIndex;

    m_taskQueue.pop();
    return taskIndex;
  }

private:


  struct Task
  {
    nvvk::SemaphoreState semaphoreState;
    uint32_t             taskIndex      = INVALID_TASK_INDEX;
    uint32_t             dependentIndex = INVALID_TASK_INDEX;
  };

  std::queue<Task> m_taskQueue;
  uint32_t         m_availableTaskBits;
};
}
