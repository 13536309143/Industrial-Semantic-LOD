//==============================================================================
// shaders/traversal/traversal_run.comp.glsl
// Main clustered LOD traversal compute shader.
// It walks instance and group hierarchy tasks, applies frustum/Hi-Z/LOD tests, emits visible clusters, and requests missing streamed groups.
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


layout(scalar, binding = BINDINGS_FRAME_UBO, set = 0) uniform frameConstantsBuffer{FrameConstants view;};


layout(scalar, binding = BINDINGS_READBACK_SSBO, set = 0) buffer readbackBuffer{Readback readback;};


layout(scalar, binding = BINDINGS_RENDERINSTANCES_SSBO, set = 0) buffer renderInstancesBuffer{RenderInstance instances[];};


layout(scalar, binding = BINDINGS_GEOMETRIES_SSBO, set = 0) buffer geometryBuffer{Geometry geometries[];};

#if USE_TWO_PASS_CULLING && TARGETS_RASTERIZATION


layout(binding = BINDINGS_HIZ_TEX)  uniform sampler2D texHizFar[2];
#else


layout(binding = BINDINGS_HIZ_TEX)  uniform sampler2D texHizFar;
#endif


layout(scalar, binding = BINDINGS_SCENEBUILDING_UBO, set = 0) uniform buildBuffer{SceneBuilding build; };


layout(scalar, binding = BINDINGS_SCENEBUILDING_SSBO, set = 0) coherent buffer buildBufferRW{volatile SceneBuilding buildRW;  };
#if USE_STREAMING


layout(scalar, binding = BINDINGS_STREAMING_UBO, set = 0) uniform streamingBuffer{SceneStreaming streaming;};


layout(scalar, binding = BINDINGS_STREAMING_SSBO, set = 0) buffer streamingBufferRW{SceneStreaming streamingRW;};
#endif


layout(local_size_x=TRAVERSAL_RUN_WORKGROUP) in;
#include "culling.glsl"
#include "traversal.glsl"


#define USE_ATOMIC_LOAD_STORE 1


// Decode one queued traversal task and resolve the instance/node/group it references.
uint setupTask(inout TraversalInfo traversalInfo, uint readIndex, uint pass)
{
  uint subCount = 0;

  bool isNode = PACKED_GET(traversalInfo.packedNode, Node_packed_isGroup) == 0;
  if (isNode) {


    subCount = PACKED_GET(traversalInfo.packedNode, Node_packed_nodeChildCountMinusOne);
  }
  else {


    subCount = PACKED_GET(traversalInfo.packedNode, Node_packed_groupClusterCountMinusOne);
  }

  return subCount + 1;
}

#if USE_CULLING && (TARGETS_RASTERIZATION || USE_FORCED_INVISIBLE_CULLING)


bool queryWasVisible(mat4x3 instanceTransform, BBox bbox, bool isNode)
{
  vec3 bboxMin = bbox.lo;
  vec3 bboxMax = bbox.hi;
  vec4 clipMin;
  vec4 clipMax;
  bool clipValid;


#if USE_TWO_PASS_CULLING
  #if USE_SEPARATE_GROUPS
    isNode = true;
  #endif


    bool useLast = !isNode || build.pass == 0;


    bool inFrustum = intersectFrustum(useLast ? build.cullViewProjMatrixLast : build.cullViewProjMatrix, bboxMin, bboxMax, instanceTransform, clipMin, clipMax, clipValid);

    bool isVisible = inFrustum &&

      (!clipValid || (intersectSize(clipMin, clipMax, 1.0) && intersectHiz(clipMin, clipMax, useLast ? 0 : 1)));

  #if !USE_SEPARATE_GROUPS


    if (!isNode && build.pass == 1)
    {
      if (isVisible) {

        isVisible = false;
      }
      else {


        inFrustum = intersectFrustum(build.cullViewProjMatrix, bboxMin, bboxMax, instanceTransform, clipMin, clipMax, clipValid);

        isVisible = inFrustum &&
          (!clipValid || (intersectSize(clipMin, clipMax, 1.0) && intersectHiz(clipMin, clipMax, 1)));
      }
    }
  #endif
#else


  bool inFrustum = intersectFrustum(build.cullViewProjMatrixLast, bboxMin, bboxMax, instanceTransform, clipMin, clipMax, clipValid);
  bool isVisible = inFrustum &&
    (!clipValid || (intersectSize(clipMin, clipMax, 1.0) && intersectHiz(clipMin, clipMax, 0)));
#endif
  return isVisible;
}
#endif


// Expand one child node or group, producing render clusters, follow-up traversal tasks, or streaming requests.
void processSubTask(const TraversalInfo subgroupTasks, uint taskID, uint taskSubID, bool isValid, uint threadReadIndex, uint pass)
{

  TraversalInfo traversalInfo;

  traversalInfo.instanceID               = subgroupShuffle(subgroupTasks.instanceID, taskID);

  traversalInfo.packedNode               = subgroupShuffle(subgroupTasks.packedNode, taskID);

  uint instanceID     = traversalInfo.instanceID;
  bool forceCluster   = false;

  bool isNode  = PACKED_GET(traversalInfo.packedNode, Node_packed_isGroup) == 0;
#if USE_SEPARATE_GROUPS
  isNode = true;
#endif

  uint geometryID   = instances[instanceID].geometryID;
  Geometry geometry = geometries[geometryID];

  TraversalMetric traversalMetric;
#if USE_CULLING && (TARGETS_RASTERIZATION || USE_FORCED_INVISIBLE_CULLING)
  BBox bbox;
#endif

  if (isNode)
  {


    uint childIndex     = taskSubID;
    uint childNodeIndex = PACKED_GET(traversalInfo.packedNode, Node_packed_nodeChildOffset) + childIndex;
    Node childNode      = geometry.nodes.d[childNodeIndex];
    traversalMetric     = childNode.traversalMetric;
  #if USE_CULLING && (TARGETS_RASTERIZATION || USE_FORCED_INVISIBLE_CULLING)
    bbox = geometry.nodeBboxes.d[childNodeIndex];
  #endif

    traversalInfo.packedNode = childNode.packed;
  }
#if !USE_SEPARATE_GROUPS
  else {

    uint clusterIndex = taskSubID;

    uint groupIndex   = PACKED_GET(traversalInfo.packedNode, Node_packed_groupIndex);

  #if USE_STREAMING


    Group_in groupRef = Group_in(geometry.streamingGroupAddresses.d[groupIndex]);
    Group group = groupRef.d;
  #else

    Group_in groupRef = Group_in(geometry.preloadedGroups.d[groupIndex]);
    Group group = groupRef.d;
  #endif
  #if USE_CULLING && (TARGETS_RASTERIZATION || USE_FORCED_INVISIBLE_CULLING)

    bbox        = Group_getClusterBBox(groupRef, clusterIndex);
  #endif

    uint32_t clusterGeneratingGroup = Group_getGeneratingGroup(groupRef, clusterIndex);
  #if USE_STREAMING
    if (clusterGeneratingGroup != SHADERIO_ORIGINAL_MESH_GROUP
        && geometry.streamingGroupAddresses.d[clusterGeneratingGroup] < STREAMING_INVALID_ADDRESS_START)
    {

traversalMetric = Group_in(geometry.streamingGroupAddresses.d[clusterGeneratingGroup]).d.traversalMetric;
    }
  #else
    if (clusterGeneratingGroup != SHADERIO_ORIGINAL_MESH_GROUP)
    {
      traversalMetric = Group_in(geometry.preloadedGroups.d[clusterGeneratingGroup]).d.traversalMetric;
}
  #endif
    else {

      traversalMetric = group.traversalMetric;
      forceCluster    = true;
    }

    traversalInfo.packedNode = group.clusterResidentID + clusterIndex;
  }
#endif


  mat4x3 worldMatrix = instances[instanceID].worldMatrix;

  float uniformScale = computeUniformScale(worldMatrix);
  float errorScale   = max(instances[instanceID].lodErrorScale, 0.01);
#if USE_CULLING && (TARGETS_RASTERIZATION || USE_FORCED_INVISIBLE_CULLING)


  isValid            = isValid && queryWasVisible(worldMatrix, bbox, isNode);
#endif

  bool traverse      = testForTraversal(mat4x3(build.traversalViewMatrix * toMat4(worldMatrix)), uniformScale, traversalMetric, errorScale);


  float lodTransitionFactor = 0.0;
  if (isNode) {


  }


  bool traverseNode  = isValid && isNode && (traverse);

  bool renderCluster = isValid && !isNode && (!traverse || forceCluster);
  bool isGroup = false;
#if USE_STREAMING || USE_SEPARATE_GROUPS
  if (traverseNode)
  {


         isGroup    = PACKED_GET(traversalInfo.packedNode, Node_packed_isGroup) != 0;

    uint groupIndex = PACKED_GET(traversalInfo.packedNode, Node_packed_groupIndex);

  #if USE_STREAMING

    if (isGroup)
    {
      uint64_t groupAddress = geometry.streamingGroupAddresses.d[groupIndex];
      if (groupAddress >= STREAMING_INVALID_ADDRESS_START) {

        traverseNode = false;
        {


          uint64_t lastRequestFrameIndex = atomicMax(geometry.streamingGroupAddresses.d[groupIndex], streaming.request.frameIndex);

          bool triggerRequest = lastRequestFrameIndex != streaming.request.frameIndex;


          uvec4 voteRequested  = subgroupBallot(triggerRequest);

          uint  countRequested = subgroupBallotBitCount(voteRequested);
          uint offsetRequested = 0;
          if (subgroupElect()) {

                 offsetRequested = atomicAdd(streamingRW.request.loadCounter, countRequested);
               }

          offsetRequested = subgroupBroadcastFirst(offsetRequested);

          offsetRequested += subgroupBallotExclusiveBitCount(voteRequested);
          if (triggerRequest && offsetRequested <= streaming.request.maxLoads) {


                 streaming.request.loadGeometryGroups.d[offsetRequested] = uvec2(geometryID, groupIndex);
               }
        }
      }
    }
  #endif
  }
#endif
#if USE_SEPARATE_GROUPS
  renderCluster = isValid && traverseNode && isGroup;
  if (renderCluster)
    traverseNode = false;
#endif


  uvec4 voteNodes = subgroupBallot(traverseNode);

  uint countNodes = subgroupBallotBitCount(voteNodes);

  uvec4 voteClusters = subgroupBallot(renderCluster);

  uint countClusters = subgroupBallotBitCount(voteClusters);
  uint offsetNodes    = 0;
  uint offsetClusters = 0;


  if (subgroupElect())
  {

    atomicAdd(buildRW.traversalTaskCounter, int(countNodes));


    offsetNodes    = atomicAdd(buildRW.traversalInfoWriteCounter, countNodes);
  #if USE_SEPARATE_GROUPS

    offsetClusters = atomicAdd(buildRW.traversalGroupCounter, countClusters);
  #else

    offsetClusters = atomicAdd(buildRW.renderClusterCounter, countClusters);
  #endif
  }

  memoryBarrierBuffer();


  offsetNodes = subgroupBroadcastFirst(offsetNodes);

  offsetNodes += subgroupBallotExclusiveBitCount(voteNodes);

  offsetClusters = subgroupBroadcastFirst(offsetClusters);

  offsetClusters += subgroupBallotExclusiveBitCount(voteClusters);

  traverseNode  = traverseNode && offsetNodes < build.maxTraversalInfos;

#if USE_SEPARATE_GROUPS
  renderCluster = renderCluster && offsetClusters < build.maxTraversalInfos;
#else
  renderCluster = renderCluster && offsetClusters < build.maxRenderClusters;
#endif

  bool doStore = traverseNode || renderCluster;
  if (doStore)
  {
    uint writeIndex          = traverseNode ? offsetNodes : offsetClusters;

    uint64s_coh writePointer = uint64s_coh(traverseNode ? uint64_t(build.traversalNodeInfos)
#if USE_SEPARATE_GROUPS

    : uint64_t(build.traversalGroupInfos)
#else

    : uint64_t(build.renderClusterInfos)
#endif
    );
  #if USE_ATOMIC_LOAD_STORE

    atomicStore(writePointer.d[writeIndex], packTraversalInfo(traversalInfo), gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsRelease);
  #else

    writePointer.d[writeIndex] = packTraversalInfo(traversalInfo);
  #endif


  }
}


struct TaskInfo {
  uint taskID;
};
shared TaskInfo s_tasks[TRAVERSAL_RUN_WORKGROUP];


void processAllSubTasks(inout TraversalInfo traversalInfo, bool threadRunnable, int threadSubCount, uint threadReadIndex, uint pass)
{


  int endOffset    = subgroupInclusiveAdd(threadSubCount);
  int startOffset  = endOffset - threadSubCount;

  int totalThreads = subgroupShuffle(endOffset, SUBGROUP_SIZE - 1);

  int totalRuns    = (totalThreads + SUBGROUP_SIZE - 1) / SUBGROUP_SIZE;
  const uint subgroupOffset = gl_SubgroupID * gl_SubgroupSize;

  bool hasTask     = threadSubCount > 0;

  uvec4 taskVote   = subgroupBallot(hasTask);

  uint taskCount   = subgroupBallotBitCount(taskVote);

  uint taskOffset  = subgroupBallotExclusiveBitCount(taskVote);


  if (hasTask) {
    s_tasks[subgroupOffset + taskOffset].taskID = gl_SubgroupInvocationID;
  }

  memoryBarrierShared();
  uint sumEnqueues = 0;

  int taskBase = -1;
  for (int r = 0; r < totalRuns; r++)
  {

    int tFirst = r * SUBGROUP_SIZE;

    int t      = tFirst + int(gl_SubgroupInvocationID);
    int  relStart      = startOffset - tFirst;
#if SUBGROUP_SIZE > 32
    uvec2 startBits    = subgroupOr(unpack32(threadRunnable && relStart >= 0 && relStart < SUBGROUP_SIZE ? (uint64_t(1) << relStart) : uint64_t(0)));
    int  task          = bitCount(startBits.x & gl_SubgroupLeMask.x) + bitCount(startBits.y & gl_SubgroupLeMask.y) + taskBase;
#else
    uint startBits     = subgroupOr(threadRunnable && relStart >= 0 && relStart < SUBGROUP_SIZE ? (1 << relStart) : 0);
    int  task          = bitCount(startBits & gl_SubgroupLeMask.x) + taskBase;
#endif
    uint taskID        = s_tasks[subgroupOffset + task].taskID;

    uint taskSubID     = t - subgroupShuffle(startOffset, taskID);

    uint taskSubCount  = subgroupShuffle(threadSubCount, taskID);
    #if 0
#else
    uint taskReadIndex = 0;
  #endif

    taskBase           = subgroupShuffle(task, SUBGROUP_SIZE - 1);
    bool taskValid     = taskSubID < taskSubCount;

    processSubTask(traversalInfo, taskID, min(taskSubID,taskSubCount-1), taskValid, taskReadIndex, pass);
  }
}


// Process the current traversal queue pass; pass 1 consumes work generated by adaptive two-pass culling.
void run()
{


  uint threadReadIndex = ~0;
  for(uint pass = 0; ; pass++)
  {

    if (subgroupAll(threadReadIndex == ~0)) {

      if (subgroupElect()){


        threadReadIndex = atomicAdd(buildRW.traversalInfoReadCounter, SUBGROUP_SIZE);
      }
      threadReadIndex = subgroupBroadcastFirst(threadReadIndex) + gl_SubgroupInvocationID;
      threadReadIndex = threadReadIndex >= build.maxTraversalInfos ? ~0 : threadReadIndex;

      if (subgroupAll(threadReadIndex == ~0)){
        break;
      }
    }
    bool threadRunnable = false;
    TraversalInfo nodeTraversalInfo;

    while(true)
    {
      if (threadReadIndex != ~0)
      {

      #if USE_ATOMIC_LOAD_STORE

        uint64_t rawValue = atomicLoad(build.traversalNodeInfos.d[threadReadIndex], gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsAcquire);
      #else


        memoryBarrierBuffer();
        uint64_t rawValue = build.traversalNodeInfos.d[threadReadIndex];
      #endif

        nodeTraversalInfo = unpackTraversalInfo(rawValue);

        threadRunnable    = nodeTraversalInfo.instanceID != ~0u && nodeTraversalInfo.packedNode != ~0u;
      }

      if (subgroupAny(threadRunnable))
        break;

      #if USE_ATOMIC_LOAD_STORE
        bool isEmpty = atomicLoad(buildRW.traversalTaskCounter, gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsAcquire) == 0;
      #else

        memoryBarrierBuffer();
        bool isEmpty = buildRW.traversalTaskCounter == 0;
      #endif
      if (subgroupAny(isEmpty))
      {
        return;
      }
    }
    if (subgroupAny(threadRunnable))
    {
#if !USE_LOW_DETAIL_FAST_PATH
      bool deferredCoarse = threadRunnable && nodeTraversalInfo.packedNode == 0xfffffffeu;
      uvec4 coarseVote = subgroupBallot(deferredCoarse);
      uint coarseCount = subgroupBallotBitCount(coarseVote);
      uint coarseOffset = 0;
      if(subgroupElect()) coarseOffset = atomicAdd(buildRW.renderClusterCounter, coarseCount);
      coarseOffset = subgroupBroadcastFirst(coarseOffset) + subgroupBallotExclusiveBitCount(coarseVote);
      if(deferredCoarse && coarseOffset < build.maxRenderClusters)
      {
        ClusterInfo info;
        info.instanceID = nodeTraversalInfo.instanceID;
        info.clusterID = geometries[instances[info.instanceID].geometryID].lowDetailClusterID;
        build.renderClusterInfos.d[coarseOffset] = info;
      }
      bool regularTask = threadRunnable && !deferredCoarse;
#else
      bool regularTask = threadRunnable;
#endif
      int threadSubCount = 0;
      if (regularTask)

      {
        threadSubCount = int(setupTask(nodeTraversalInfo, threadReadIndex, pass));
      }


      if(subgroupAny(regularTask))
        processAllSubTasks(nodeTraversalInfo, regularTask, threadSubCount, threadReadIndex, pass);
    #if USE_TWO_PASS_CULLING && TARGETS_RASTERIZATION

      if (build.pass == 0 && threadRunnable) {
        build.traversalNodeInfos.d[threadReadIndex] = uint64_t(packUint2x32(uvec2(~0, ~0)));
      }
    #endif

      uint numRunnable = subgroupBallotBitCount(subgroupBallot(threadRunnable));
      if (subgroupElect()) {
        atomicAdd(buildRW.traversalTaskCounter, -int(numRunnable));
      }
      if (threadRunnable) {

        threadReadIndex = ~0;
      }
    }
  }
}


void main()
{

  run();

#if USE_SEPARATE_GROUPS

  uint threadID = getGlobalInvocationIndex(gl_GlobalInvocationID);
  if (threadID == 0) {


    uint groupCount = atomicAdd(buildRW.traversalGroupCounter,0);

    groupCount = min(groupCount,build.maxTraversalInfos);
    groupCount = min(groupCount,build.maxTraversalInfos);
#if PARALLEL_GROUPS
    uint workGroupCount = (groupCount * uint(GROUP_CLUSTER_LANES) + TRAVERSAL_GROUPS_WORKGROUP - 1) / TRAVERSAL_GROUPS_WORKGROUP;
#else
    uint workGroupCount = (groupCount + TRAVERSAL_GROUPS_WORKGROUP - 1) / TRAVERSAL_GROUPS_WORKGROUP;
#endif
  #if USE_16BIT_DISPATCH

    uvec3 grid = fit16bitLaunchGrid(workGroupCount);
    buildRW.indirectDispatchGroups.gridX = grid.x;
    buildRW.indirectDispatchGroups.gridY = grid.y;
    buildRW.indirectDispatchGroups.gridZ = grid.z;
  #else
    buildRW.indirectDispatchGroups.gridX = workGroupCount;
  #endif
  }
#endif
}
