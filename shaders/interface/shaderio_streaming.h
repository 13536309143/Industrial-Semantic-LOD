//==============================================================================
// shaders/interface/shaderio_streaming.h
// Streaming ABI shared between CPU task scheduling and GPU request/update passes.
// Resident IDs, invalid-address sentinels, and patch records must remain stable across traversal, update, and render stages.
//==============================================================================
#ifndef _SHADERIO_STREAMING_H_
#define _SHADERIO_STREAMING_H_

#include "shaderio_scene.h"

#ifdef __cplusplus
namespace shaderio {
using namespace glm;
#endif

#define STREAMING_INVALID_ADDRESS_START (uint64_t(1) << 63)
#define STREAMING_DEBUG_ADDRESSES 0
#define STREAMING_DEBUG_ALWAYS_BUILD_FREEGAPS 0

struct StreamingRequest
{
  uint32_t maxLoads;
  uint32_t maxUnloads;
  uint32_t loadCounter;
  uint32_t unloadCounter;
#ifdef __cplusplus
  union
  {
    uint64_t frameIndex;
    uint32_t frameIndexU32[2];
  };
#else
  uint64_t frameIndex;
#endif
  uint32_t taskIndex;
  uint32_t errorUpdate;
  uint32_t errorAgeFilter;

  BUFFER_REF(uvec2s_inout) loadGeometryGroups;
  BUFFER_REF(uvec2s_inout) unloadGeometryGroups;
};

struct StreamingPatch
{
  uint32_t geometryID;
  uint32_t groupID;
  uint64_t groupAddress;

  uint16_t clusterCount;
  uint16_t lodLevel;
  uint32_t groupResidentID;
  uint32_t clusterResidentID;
};

BUFFER_REF_DECLARE_ARRAY(StreamingPatchs_in, StreamingPatch, , 16);

struct StreamingUpdate
{
  uint32_t patchUnloadGroupsCount;
  uint32_t patchGroupsCount;

  uint32_t loadActiveGroupsOffset;
  uint32_t loadActiveClustersOffset;

  uint32_t taskIndex;
  uint32_t frameIndex;

  BUFFER_REF(StreamingPatchs_in) patches;
};

struct StreamingGroup
{
  uint32_t geometryID;
  uint16_t lodLevel;
  uint16_t age;
  BUFFER_REF(Group_in) group;
};

BUFFER_REF_DECLARE_ARRAY(StreamingGroup_inout, StreamingGroup, , 16);

struct StreamingResident
{
  uint32_t activeGroupsCount;
  uint32_t activeClustersCount;

  BUFFER_REF(uint32s_in) activeGroups;
  BUFFER_REF(StreamingGroup_inout) groups;
  BUFFER_REF(uint32s_inout) groupIDs;
  BUFFER_REF(uint64s_inout) clusters;

  uint32_t taskIndex;
  uint32_t frameIndex;
};

struct SceneStreaming
{
  int32_t ageThreshold;
  uint32_t frameIndex;

  StreamingResident resident;
  StreamingUpdate   update;
  StreamingRequest  request;
};

#ifdef __cplusplus
}
#endif

#endif
