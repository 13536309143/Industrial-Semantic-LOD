//==============================================================================
// shaders/traversal/traversal_init.comp.glsl
// Seeds traversal queues from the visible render-instance list.
// The pass writes initial root tasks and resets counters before the main traversal kernel expands the hierarchy.
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


#if USE_TWO_PASS_CULLING


layout(binding = BINDINGS_HIZ_TEX)  uniform sampler2D texHizFar[2];
#else


layout(binding = BINDINGS_HIZ_TEX)  uniform sampler2D texHizFar;
#endif


layout(scalar, binding = BINDINGS_SCENEBUILDING_UBO, set = 0) uniform buildBuffer
{
  SceneBuilding build;
};


layout(scalar, binding = BINDINGS_SCENEBUILDING_SSBO, set = 0) buffer buildBufferRW
{
  SceneBuilding buildRW;
};


layout(local_size_x=TRAVERSAL_INIT_WORKGROUP) in;

#include "culling.glsl"
#include "traversal.glsl"


void main()
{

  uint instanceID = getGlobalInvocationIndex(gl_GlobalInvocationID);
#if USE_COMPACT_CANDIDATES
  bool compactList = build.numCandidateRanges != 0u;
  uint count = compactList ? build.candidateControl.d[0].instanceCount : build.numRenderInstances;
  bool isValid = instanceID < count;
  uint instanceLoad = compactList ? build.candidateInstances.d[min(instanceID,max(count,1u)-1u)]
                                 : min(instanceID,build.numRenderInstances-1u);
  instanceID=instanceLoad;
  if(compactList && !isValid) return;
#else
  uint instanceLoad = min(build.numRenderInstances-1u, instanceID);
  bool isValid = instanceID == instanceLoad;
#endif
#if USE_RENDER_STATS
  if(isValid) atomicAdd(readback.numInitializedInstances,1u);
#endif

#if USE_SORTING
  instanceLoad = build.instanceSortValues.d[instanceLoad];
  instanceID   = instanceLoad;
#endif

#if USE_ASSEMBLY_EARLY_EXIT
  // Invalid lanes must not participate in queue emission or touch scene memory.
  if(!isValid) return;
#endif
  RenderInstance instance = instances[instanceLoad];

  vec4 clipMin;
  vec4 clipMax;
  bool clipValid;

  bool assemblyVisible = true;
  bool assemblyForceLowDetail = false;

  if (USE_ASSEMBLY_PREFILTER != 0 && isValid && instance.assemblyID != SHADERIO_INVALID_ASSEMBLY && instance.assemblyID < build.numAssemblyNodes
#if USE_COMPACT_CANDIDATES && QUALITY_BOUNDED_ASSEMBLY
      && !compactList
#endif
     )
  {
    AssemblyState assemblyState = build.assemblyStates.d[instance.assemblyID];
    assemblyVisible = (assemblyState.flags & SHADERIO_ASSEMBLY_VISIBLE_BIT) != 0;
    assemblyForceLowDetail = (assemblyState.flags & SHADERIO_ASSEMBLY_LOD_COARSE_BIT) != 0;
  }

#if USE_ASSEMBLY_EARLY_EXIT && USE_CULLING && (TARGETS_RASTERIZATION || USE_FORCED_INVISIBLE_CULLING)
  // Reject before fetching Geometry and before traversal queue operations.
  // Pass zero must clear visibility so stale state cannot suppress pass one.
  if(!assemblyVisible)
  {
#if USE_RENDER_STATS
    atomicAdd(readback.numAssemblyRejectedInstances, 1u);
#endif
#if USE_TWO_PASS_CULLING && TARGETS_RASTERIZATION
    if(build.pass == 0) build.instanceVisibility.d[instanceID] = uint8_t(0);
#endif
    return;
  }
#endif
  uint geometryID = instance.geometryID;
  Geometry geometry = geometries[geometryID];

#if USE_TWO_PASS_CULLING && TARGETS_RASTERIZATION


  bool inFrustum = false;
  bool isVisible = false;
  if (assemblyVisible) {
    inFrustum = intersectFrustum( build.pass == 0 ? build.cullViewProjMatrixLast : build.cullViewProjMatrix, geometry.bbox.lo, geometry.bbox.hi, instance.worldMatrix, clipMin, clipMax, clipValid);
    isVisible = inFrustum && (!clipValid || (intersectSize(clipMin, clipMax, 1.0) && intersectHiz(clipMin, clipMax, build.pass)));
  }


  if (build.pass == 1 && isVisible && clipValid && !intersectSize(clipMin, clipMax, 8.0) && ((uint(build.instanceVisibility.d[instanceLoad]) & INSTANCE_VISIBLE_BIT) != 0)) {
    isVisible = false;
  }


#else

  bool inFrustum = false;
  bool isVisible = false;
  if (assemblyVisible) {
    inFrustum = intersectFrustum(build.cullViewProjMatrixLast, geometry.bbox.lo, geometry.bbox.hi, instance.worldMatrix, clipMin, clipMax, clipValid);
    isVisible = inFrustum && (!clipValid || (intersectSize(clipMin, clipMax, 1.0) && intersectHiz(clipMin, clipMax, 0)));
  }
#endif

  uint visibilityState = isVisible ? INSTANCE_VISIBLE_BIT : 0;
#if USE_RENDER_STATS
  if(isValid && !assemblyVisible) atomicAdd(readback.numAssemblyRejectedInstances, 1u);
#endif

  bool isRenderable = isValid
  #if USE_CULLING && (TARGETS_RASTERIZATION || USE_FORCED_INVISIBLE_CULLING)
    && isVisible
  #endif
    ;

  bool traverseRootNode = isRenderable;
  bool coarseSelected = false;

  if (isRenderable)
  {


    uint rootNodePacked = geometry.nodes.d[0].packed;


    uint childOffset        = PACKED_GET(rootNodePacked, Node_packed_nodeChildOffset);

    uint childCountMinusOne = PACKED_GET(rootNodePacked, Node_packed_nodeChildCountMinusOne);


    uint childNodeIndex     = (childCountMinusOne > 1 ? (childCountMinusOne - 1) : 0);
    Node childNode          = geometry.nodes.d[childOffset + childNodeIndex];
    TraversalMetric traversalMetric = childNode.traversalMetric;

    mat4x3 worldMatrix = instances[instanceID].worldMatrix;

    float uniformScale = computeUniformScale(worldMatrix);
    float errorScale   = max(instance.lodErrorScale, 0.01);


    mat4 transform = build.traversalViewMatrix * toMat4(worldMatrix);


    bool errorAllowsCoarse = !testForTraversal(mat4x3(transform), uniformScale, traversalMetric, errorScale);
#if QUALITY_BOUNDED_ASSEMBLY
    // A small projected assembly must not override the geometric error budget.
    bool selectCoarse = errorAllowsCoarse;
#else
    bool selectCoarse = assemblyForceLowDetail || errorAllowsCoarse;
#endif
    LodLevel coarsest=geometry.lodLevels.d[geometry.lodLevelsCount-1];
    selectCoarse=selectCoarse && coarsest.groupCount==1 && coarsest.clusterCount==1;
    coarseSelected = selectCoarse;
    if (selectCoarse)
    {
#if USE_RENDER_STATS
      atomicAdd(readback.numLowDetailInstances, 1u);
#endif
    }
    if (selectCoarse && USE_LOW_DETAIL_FAST_PATH != 0)
    {

    #if TARGETS_RASTERIZATION


      uvec4 voteClusters = subgroupBallot(true);

      uint offsetClusters = 0;
      if (subgroupElect())
      {
        offsetClusters = atomicAdd(buildRW.renderClusterCounter, int(subgroupBallotBitCount(voteClusters)));
      }


      offsetClusters = subgroupBroadcastFirst(offsetClusters);

      offsetClusters += subgroupBallotExclusiveBitCount(voteClusters);

      if (offsetClusters < build.maxRenderClusters)
      {
        ClusterInfo clusterInfo;
        clusterInfo.instanceID = instanceID;
        clusterInfo.clusterID  = geometry.lowDetailClusterID;
        build.renderClusterInfos.d[offsetClusters] = clusterInfo;
      }
    #endif


      traverseRootNode = false;
    }
  }


  uvec4 voteNodes = subgroupBallot(traverseRootNode);

  uint offsetNodes = 0;
  if (subgroupElect())
  {
    offsetNodes = atomicAdd(buildRW.traversalTaskCounter, int(subgroupBallotBitCount(voteNodes)));
  }


  offsetNodes = subgroupBroadcastFirst(offsetNodes);

  offsetNodes += subgroupBallotExclusiveBitCount(voteNodes);

  if (traverseRootNode && offsetNodes < build.maxTraversalInfos)
  {
    uint rootNodePacked = geometry.nodes.d[0].packed;

#if !USE_LOW_DETAIL_FAST_PATH
    // Preserve exactly the selected coarse geometry, but process it through
    // the ordinary task queue instead of emitting it during initialization.
    if(coarseSelected) rootNodePacked = 0xfffffffeu;
#endif
#if USE_RENDER_STATS
    atomicAdd(readback.numQueuedRootInstances, 1u);
#endif

    TraversalInfo traversalInfo;
    traversalInfo.instanceID = instanceID;
    traversalInfo.packedNode = rootNodePacked;


    build.traversalNodeInfos.d[offsetNodes] = packTraversalInfo(traversalInfo);
  }
#if USE_TWO_PASS_CULLING && TARGETS_RASTERIZATION
  if (build.pass == 0 && isValid) {

    build.instanceVisibility.d[instanceID]                        = uint8_t(visibilityState);
  }
#endif
}
