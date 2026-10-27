//==============================================================================
// shaders/build/build_setup.comp.glsl
// Initializes per-frame scene-building counters and indirect dispatch arguments.
// This pass decides whether the second traversal/culling pass should run and prepares render or traversal work queues.
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
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_clustered : require
#extension GL_KHR_shader_subgroup_arithmetic : require


#include "shaderio.h"


layout(push_constant) uniform pushData
{
  uint setup;
} push;


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


layout(binding = BINDINGS_HIZ_TEX)  uniform sampler2D texHizFar;


layout(scalar, binding = BINDINGS_SCENEBUILDING_UBO, set = 0) uniform buildBuffer
{
  SceneBuilding build;
};


layout(scalar, binding = BINDINGS_SCENEBUILDING_SSBO, set = 0) coherent buffer buildBufferRW
{
  SceneBuilding buildRW;
};


layout(local_size_x=1) in;


#ifndef MESHSHADER_BBOX_COUNT


#define MESHSHADER_BBOX_COUNT 8
#endif


#if USE_TWO_PASS_CULLING


void setupSecondPass()
{

  buildRW.pass = 1;
  buildRW.traversalTaskCounter = 0;
  buildRW.traversalGroupCounter = 0;
  buildRW.renderClusterCounter = 0;
  buildRW.traversalInfoReadCounter = 0;
}
#endif


void main()
{


  if (push.setup == BUILD_SETUP_TRAVERSAL_RUN)
  {

    int traversalTaskCounter = min(buildRW.traversalTaskCounter, int(build.maxTraversalInfos));
    buildRW.traversalTaskCounter = traversalTaskCounter;


    buildRW.traversalInfoWriteCounter = uint(traversalTaskCounter);
  }
#if TARGETS_RASTERIZATION
  else if (push.setup == BUILD_SETUP_DRAW)
  {

    uint renderClusterCounter = buildRW.renderClusterCounter;


    uint numRenderedClusters = min(renderClusterCounter, build.maxRenderClusters);
  #if USE_EXT_MESH_SHADER

    uvec3 grid = fit16bitLaunchGrid(numRenderedClusters);
    buildRW.indirectDrawClustersEXT.gridX = grid.x;
    buildRW.indirectDrawClustersEXT.gridY = grid.y;
    buildRW.indirectDrawClustersEXT.gridZ = grid.z;

    grid = fit16bitLaunchGrid((numRenderedClusters + MESHSHADER_BBOX_COUNT - 1) / MESHSHADER_BBOX_COUNT);
    buildRW.indirectDrawClusterBoxesEXT.gridX = grid.x;
    buildRW.indirectDrawClusterBoxesEXT.gridY = grid.y;
    buildRW.indirectDrawClusterBoxesEXT.gridZ = grid.z;
  #else
    buildRW.indirectDrawClustersNV.count = numRenderedClusters;
    buildRW.indirectDrawClustersNV.first = 0;

    buildRW.indirectDrawClusterBoxesNV.count = (numRenderedClusters + MESHSHADER_BBOX_COUNT - 1) / MESHSHADER_BBOX_COUNT;
    buildRW.indirectDrawClusterBoxesNV.first = 0;
  #endif
    buildRW.numRenderedClusters = numRenderedClusters;


    atomicMax(readback.numRenderClusters, renderClusterCounter);
  #if USE_SEPARATE_GROUPS

    atomicMax(readback.numTraversalTasks, max(buildRW.traversalInfoWriteCounter, buildRW.traversalGroupCounter));
  #else


    atomicMax(readback.numTraversalTasks, buildRW.traversalInfoWriteCounter);
  #endif

  #if USE_RENDER_STATS

    readback.numRenderedClusters += numRenderedClusters;
    readback.numTraversedTasks   += buildRW.traversalInfoWriteCounter;
    if(build.pass == 0)
    {
      readback.numRenderedClustersPass0 += numRenderedClusters;
      readback.numTraversedTasksPass0 += buildRW.traversalInfoWriteCounter;
    }
    else
    {
      readback.numRenderedClustersPass1 += numRenderedClusters;
      readback.numTraversedTasksPass1 += buildRW.traversalInfoWriteCounter;
    }
  #endif
    readback.twoPassCullingActive = build.twoPassCullingActive;
    readback.adaptiveTemporalBypass = build.adaptiveTemporalBypass;
    readback.adaptiveTriggerReason = build.adaptiveTriggerReason;
    readback.adaptiveMotionPixels = build.adaptiveMotionPixels;
    readback.adaptiveAccumulatedPixels = build.adaptiveAccumulatedPixels;
    readback.adaptiveRefreshAge = build.adaptiveRefreshAge;

  #if USE_TWO_PASS_CULLING

    if(build.twoPassCullingActive != 0)
    {
      setupSecondPass();
    }
  #endif
  }
#endif
}
