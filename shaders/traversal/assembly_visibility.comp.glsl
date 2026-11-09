//==============================================================================
// shaders/traversal/assembly_visibility.comp.glsl
// Classifies assembly visibility using conservative screen-space and Hi-Z tests.
// Results allow traversal to skip or prioritize groups when many instances share the same assembly.
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

#include "shaderio.h"

layout(scalar, binding = BINDINGS_FRAME_UBO, set = 0) uniform frameConstantsBuffer
{
  FrameConstants view;
};

#if USE_TWO_PASS_CULLING
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

layout(local_size_x = ASSEMBLY_VISIBILITY_WORKGROUP) in;

#include "culling.glsl"

float computeScreenPixels(vec4 clipMin, vec4 clipMax, bool clipValid)
{
  if(!clipValid)
  {
    return 3.402823466e+38f;
  }

  vec2 rect = (clipMax.xy - clipMin.xy) * 0.5 * view.viewportf.xy;
  return max(rect.x, rect.y);
}

void main()
{
  uint assemblyID = getGlobalInvocationIndex(gl_GlobalInvocationID);
  if(assemblyID >= build.numAssemblyNodes)
  {
    return;
  }

  AssemblyNode assembly = build.assemblyNodes.d[assemblyID];
  mat4x3 worldIdentity = mat4x3(vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), vec3(0.0, 0.0, 1.0), vec3(0.0, 0.0, 0.0));

  vec4 clipMin;
  vec4 clipMax;
  bool clipValid;

#if USE_CULLING && (TARGETS_RASTERIZATION || USE_FORCED_INVISIBLE_CULLING)
  #if USE_TWO_PASS_CULLING && TARGETS_RASTERIZATION
    uint previousFlags = 0u;
    if(build.pass == 1)
    {
      previousFlags = build.assemblyStates.d[assemblyID].flags;
    }
    bool inFrustum = intersectFrustum(build.pass == 0 ? build.cullViewProjMatrixLast : build.cullViewProjMatrix,
                                      assembly.bbox.lo, assembly.bbox.hi, worldIdentity, clipMin, clipMax, clipValid);
    bool isVisible = inFrustum && (!clipValid || (intersectSize(clipMin, clipMax, 1.0) && intersectHiz(clipMin, clipMax, build.pass)));

    if(build.pass == 1 && isVisible && clipValid && !intersectSize(clipMin, clipMax, 8.0)
       && ((previousFlags & SHADERIO_ASSEMBLY_VISIBLE_BIT) != 0))
    {
      isVisible = false;
    }
  #else
    bool inFrustum = intersectFrustum(build.cullViewProjMatrixLast, assembly.bbox.lo, assembly.bbox.hi, worldIdentity, clipMin, clipMax, clipValid);
    bool isVisible = inFrustum && (!clipValid || (intersectSize(clipMin, clipMax, 1.0) && intersectHiz(clipMin, clipMax, 0)));
  #endif
#else
  bool isVisible = true;
  clipValid = true;
  clipMin = vec4(-1.0);
  clipMax = vec4(1.0);
#endif

  float screenPixels = computeScreenPixels(clipMin, clipMax, clipValid);
  float extentRadius = length(assembly.bbox.hi - assembly.bbox.lo) * 0.5;
  vec3 center = (assembly.bbox.lo + assembly.bbox.hi) * 0.5;
  vec3 viewCenter = vec3(build.traversalViewMatrix * vec4(center, 1.0));
  float errorDistance = max(view.nearPlane, length(viewCenter) - extentRadius);
  float errorOverDistance = extentRadius / errorDistance;

  uint flags = isVisible ? SHADERIO_ASSEMBLY_VISIBLE_BIT : 0u;
  if(isVisible && build.assemblyLodPixelThreshold > 0.0 && screenPixels <= build.assemblyLodPixelThreshold)
  {
    flags |= SHADERIO_ASSEMBLY_LOD_COARSE_BIT;
  }

  AssemblyState state;
  state.flags = flags;
  state.screenPixels = screenPixels;
  state.errorOverDistance = errorOverDistance;
  state.reserved = 0u;
  build.assemblyStates.d[assemblyID] = state;
}
