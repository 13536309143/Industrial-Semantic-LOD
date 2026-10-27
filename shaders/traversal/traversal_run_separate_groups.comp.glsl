//==============================================================================
// shaders/traversal/traversal_run_separate_groups.comp.glsl
// Traversal variant that processes group visibility separately from node traversal.
// This path keeps group-level work explicit for experiments with queue pressure and streaming behavior.
//==============================================================================
#version 460

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_buffer_reference2 : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_shader_atomic_int64 : enable
#extension GL_EXT_control_flow_attributes : require
#extension GL_KHR_shader_subgroup_vote : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_clustered : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_memory_scope_semantics : require


#include "shaderio.h"


layout(scalar, binding = BINDINGS_FRAME_UBO, set = 0) uniform frameConstantsBuffer
{
  FrameConstants view;
};


layout(scalar, binding = BINDINGS_READBACK_SSBO, set = 0) buffer readbackBuffer
{
  Readback readback;
};


layout(scalar, binding = BINDINGS_RENDERINSTANCES_SSBO, set = 0) buffer renderInstancesBuffer
{
  RenderInstance instances[];
};


layout(scalar, binding = BINDINGS_GEOMETRIES_SSBO, set = 0) buffer geometryBuffer
{
  Geometry geometries[];
};

#if USE_TWO_PASS_CULLING && TARGETS_RASTERIZATION


layout(binding = BINDINGS_HIZ_TEX) uniform sampler2D texHizFar[2];
#else


layout(binding = BINDINGS_HIZ_TEX) uniform sampler2D texHizFar;
#endif


layout(scalar, binding = BINDINGS_SCENEBUILDING_UBO, set = 0) uniform buildBuffer
{
  SceneBuilding build;
};


layout(scalar, binding = BINDINGS_SCENEBUILDING_SSBO, set = 0) buffer buildBufferRW
{
  SceneBuilding buildRW;
};

#if USE_STREAMING


layout(scalar, binding = BINDINGS_STREAMING_UBO, set = 0) uniform streamingBuffer
{
  SceneStreaming streaming;
};


layout(scalar, binding = BINDINGS_STREAMING_SSBO, set = 0) buffer streamingBufferRW
{
  SceneStreaming streamingRW;
};
#endif


layout(local_size_x = TRAVERSAL_GROUPS_WORKGROUP) in;

#include "culling.glsl"
#include "traversal.glsl"


#define USE_ATOMIC_LOAD_STORE 1

#if USE_CULLING && (TARGETS_RASTERIZATION || USE_FORCED_INVISIBLE_CULLING)


bool intersectSize(vec4 clipMin, vec4 clipMax, float threshold, float scale)
{
  vec2 rect          = (clipMax.xy - clipMin.xy) * 0.5 * scale * view.viewportf.xy;

  vec2 clipThreshold = vec2(threshold);
  return any(greaterThan(rect, clipThreshold));
}


bool queryWasVisible(mat4x3 instanceTransform, BBox bbox, out vec4 outClipMin, out vec4 outClipMax, out bool outClipValid)
{
  vec3 bboxMin = bbox.lo;
  vec3 bboxMax = bbox.hi;
  vec4 clipMin;
  vec4 clipMax;
  bool clipValid;
  bool useOcclusion = true;


  bool inFrustum = intersectFrustum(build.cullViewProjMatrixLast, bboxMin, bboxMax, instanceTransform, clipMin, clipMax, clipValid);
  bool isVisible = inFrustum
                   && (!useOcclusion || !clipValid || (intersectSize(clipMin, clipMax, 1.0) && intersectHiz(clipMin, clipMax, 0)));

#if USE_TWO_PASS_CULLING
  if(build.pass == 1)
  {
    if(isVisible)
    {
      isVisible = false;
    }
    else
    {

      inFrustum = intersectFrustum(build.cullViewProjMatrix, bboxMin, bboxMax, instanceTransform, clipMin, clipMax, clipValid);
      isVisible = inFrustum && (!clipValid || (intersectSize(clipMin, clipMax, 1.0) && intersectHiz(clipMin, clipMax, 1)));
    }
  }
#endif

  outClipMin   = clipMin;
  outClipMax   = clipMax;
  outClipValid = clipValid;
  return isVisible;
}
#endif


void main()
{

  uint invocation = getGlobalInvocationIndex(gl_GlobalInvocationID);
#if PARALLEL_GROUPS
  // Consecutive lanes process consecutive clusters of the same group.
  // Stride within each group; several small groups can share a subgroup.
  uint threadReadIndex = invocation / uint(GROUP_CLUSTER_LANES);
  uint firstCluster = invocation % uint(GROUP_CLUSTER_LANES);
  const uint clusterStride = uint(GROUP_CLUSTER_LANES);
#else
  uint threadReadIndex = invocation;
  uint firstCluster = 0u;
  const uint clusterStride = 1u;
#endif
  if(threadReadIndex >= min(build.traversalGroupCounter, build.maxTraversalInfos))
  {
    return;
  }


  TraversalInfo traversalInfo = unpackTraversalInfo(build.traversalGroupInfos.d[threadReadIndex]);
  uint          instanceID    = traversalInfo.instanceID;

  uint          groupIndex    = PACKED_GET(traversalInfo.packedNode, Node_packed_groupIndex);
  uint          groupClusterCount = PACKED_GET(traversalInfo.packedNode, Node_packed_groupClusterCountMinusOne) + 1;

  uint     geometryID = instances[instanceID].geometryID;
  Geometry geometry   = geometries[geometryID];

  mat4x3 worldMatrix     = instances[instanceID].worldMatrix;

  float  uniformScale    = computeUniformScale(worldMatrix);
  float  errorScale      = max(instances[instanceID].lodErrorScale, 0.01);
  mat4x3 traversalMatrix = mat4x3(build.traversalViewMatrix * toMat4(worldMatrix));

#if USE_STREAMING

  Group_in groupRef = Group_in(geometry.streamingGroupAddresses.d[groupIndex]);
  Group    group    = groupRef.d;

  streaming.resident.groups.d[group.residentID].age = uint16_t(0);
#else

  Group_in groupRef = Group_in(geometry.preloadedGroups.d[groupIndex]);
  Group    group    = groupRef.d;
#endif

  for(uint clusterIndex = firstCluster; clusterIndex < groupClusterCount; clusterIndex += clusterStride)
  {
    bool            forceCluster   = false;
    bool            isValid        = true;
    TraversalMetric traversalMetric;
#if USE_CULLING && (TARGETS_RASTERIZATION || USE_FORCED_INVISIBLE_CULLING)

    BBox bbox = Group_getClusterBBox(groupRef, clusterIndex);
#endif


    uint32_t clusterGeneratingGroup = Group_getGeneratingGroup(groupRef, clusterIndex);
#if USE_STREAMING
    if(clusterGeneratingGroup != SHADERIO_ORIGINAL_MESH_GROUP
       && geometry.streamingGroupAddresses.d[clusterGeneratingGroup] < STREAMING_INVALID_ADDRESS_START)
    {
      traversalMetric = Group_in(geometry.streamingGroupAddresses.d[clusterGeneratingGroup]).d.traversalMetric;
    }
#else
    if(clusterGeneratingGroup != SHADERIO_ORIGINAL_MESH_GROUP)
    {
      traversalMetric = Group_in(geometry.preloadedGroups.d[clusterGeneratingGroup]).d.traversalMetric;
    }
#endif
    else
    {
      traversalMetric = group.traversalMetric;
      forceCluster    = true;
    }

    uint clusterID = group.clusterResidentID + clusterIndex;
    traversalInfo.packedNode = clusterID;

#if USE_CULLING && (TARGETS_RASTERIZATION || USE_FORCED_INVISIBLE_CULLING)
    vec4 clipMin;
    vec4 clipMax;
    bool clipValid;

    isValid = isValid && queryWasVisible(worldMatrix, bbox, clipMin, clipMax, clipValid);
#endif


    bool traverse      = testForTraversal(traversalMatrix, uniformScale, traversalMetric, errorScale);
    bool renderCluster = isValid && (!traverse || forceCluster);

    uvec4 voteClusters  = subgroupBallot(renderCluster);

    uint  countClusters = subgroupBallotBitCount(voteClusters);
    uint  offsetClusters = 0;

    if(subgroupElect())
    {

      offsetClusters = atomicAdd(buildRW.renderClusterCounter, countClusters);
    }


    offsetClusters = subgroupBroadcastFirst(offsetClusters);

    offsetClusters += subgroupBallotExclusiveBitCount(voteClusters);
    renderCluster = renderCluster && offsetClusters < build.maxRenderClusters;

    if(renderCluster)
    {
      uint writeIndex          = offsetClusters;
      uint64s_coh writePointer = uint64s_coh(uint64_t(build.renderClusterInfos));

#if USE_ATOMIC_LOAD_STORE
      atomicStore(writePointer.d[writeIndex], packTraversalInfo(traversalInfo), gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsRelease);
#else

      writePointer.d[writeIndex] = packTraversalInfo(traversalInfo);
#endif

      memoryBarrierBuffer();
    }
  }
}
