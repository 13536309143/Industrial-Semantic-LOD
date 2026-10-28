//==============================================================================
// shaders/interface/shaderio_building.h
// Per-frame scene-building ABI for traversal queues, cluster lists, visibility state, and indirect dispatch metadata.
// Compute passes update these fields atomically before render shaders consume them in the same frame.
//==============================================================================
#ifndef _SHADERIO_BUILDING_H_
#define _SHADERIO_BUILDING_H_

#include "shaderio_streaming.h"

#define TRAVERSAL_INVALID_LOD_LEVEL 0xFF

#ifdef __cplusplus
namespace shaderio {
using namespace glm;
#else
#define INSTANCE_VISIBLE_BIT 1
#define INSTANCE_USES_MERGED_BIT 2
#endif

struct TraversalInfo
{
  uint32_t instanceID;
  uint32_t packedNode;
};

struct ClusterInfo
{
  uint32_t instanceID;
  uint32_t clusterID;
};

BUFFER_REF_DECLARE_ARRAY(ClusterInfos_inout, ClusterInfo, , 8);

// Each live instance occurs exactly once in the member array. Large assembly
// buckets are split into bounded chunks; only their first chunk writes state.
struct AssemblyCandidateRange
{
  uint32_t assemblyID;
  uint32_t firstMember;
  uint32_t memberCount;
  uint32_t writeState;
};
BUFFER_REF_DECLARE_ARRAY(AssemblyCandidateRanges_in, AssemblyCandidateRange, readonly, 4);

struct CandidateControl
{
  uint32_t instanceCount;
  DispatchIndirectCommand dispatch;
};
BUFFER_REF_DECLARE_ARRAY(CandidateControls_inout, CandidateControl, , 4);

struct SceneBuilding
{
  mat4 traversalViewMatrix;
  mat4 cullViewProjMatrix;
  mat4 cullViewProjMatrixLast;

  BUFFER_REF(AssemblyCandidateRanges_in) candidateRanges;
  BUFFER_REF(uint32s_inout) candidateMembers;
  BUFFER_REF(uint32s_inout) candidateInstances;
  uint32_t numCandidateRanges;
  uint32_t candidatePadding;
  BUFFER_REF(CandidateControls_inout) candidateControl;
  uint32_t pass;
  uint32_t frameIndex;
  uint32_t twoPassCullingActive;
  uint32_t adaptiveTemporalBypass;
  uint32_t adaptiveTriggerReason;
  float adaptiveMotionPixels;
  float adaptiveAccumulatedPixels;
  uint32_t adaptiveRefreshAge;

  uint32_t numGeometries;
  uint32_t numRenderInstances;
  uint32_t maxRenderClusters;
  uint32_t maxTraversalInfos;
  uint32_t numAssemblyNodes;
  uint32_t assemblyCullingMinInstances;

  float errorOverDistanceThreshold;
  float assemblyLodPixelThreshold;
  float culledErrorScale;

  uint32_t renderClusterCounter;

  int32_t  traversalTaskCounter;
  uint32_t traversalInfoReadCounter;
  uint32_t traversalInfoWriteCounter;

  uint32_t traversalGroupCounter;

  BUFFER_REF(uint64s_coh_volatile) traversalNodeInfos;
  BUFFER_REF(uint64s_coh_volatile) traversalGroupInfos;
  BUFFER_REF(ClusterInfos_inout) renderClusterInfos;

  DispatchIndirectCommand indirectDispatchGroups;

  DrawMeshTasksIndirectCommandNV indirectDrawClustersNV;
  DrawMeshTasksIndirectCommandNV indirectDrawClusterBoxesNV;

  DrawMeshTasksIndirectCommandEXT indirectDrawClustersEXT;
  DrawMeshTasksIndirectCommandEXT indirectDrawClusterBoxesEXT;
  uint32_t                       numRenderedClusters;

  BUFFER_REF(uint8s_inout) instanceVisibility;

  BUFFER_REF(AssemblyNodes_in) assemblyNodes;
  BUFFER_REF(AssemblyStates_inout) assemblyStates;

  BUFFER_REF(uint32s_inout) instanceSortValues;
  BUFFER_REF(uint32s_inout) instanceSortKeys;
};

#ifdef __cplusplus
}
#endif

#endif
