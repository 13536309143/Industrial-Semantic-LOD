//==============================================================================
// shaders/render/clusters.mesh.glsl
// Mesh shader that expands visible cluster records into rasterized triangles.
// It resolves group payload addresses, fetches compact local indices and attributes, and emits interpolation data for shading and picking.
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
#extension GL_KHR_shader_subgroup_ballot : require

#if USE_EXT_MESH_SHADER
#extension GL_EXT_mesh_shader : require
#else
#extension GL_NV_mesh_shader : require
#endif
#extension GL_EXT_control_flow_attributes : require


#include "shaderio.h"


layout(push_constant) uniform pushData
{
  uint instanceID;
}
push;


layout(scalar, binding = BINDINGS_FRAME_UBO, set = 0) uniform frameConstantsBuffer
{
  FrameConstants view;
};

#include "culling_mesh.inc"


layout(scalar,binding=BINDINGS_READBACK_SSBO,set=0) buffer readbackBuffer
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

#if !USE_DEPTH_ONLY


layout(location = 0) out Interpolants
{
  flat uint clusterID;
  flat uint instanceID;
#if ALLOW_SHADING
  vec3      wPos;
#endif
}
OUT[];
#if ALLOW_SHADING && (ALLOW_VERTEX_NORMALS || ALLOW_VERTEX_TEXCOORDS)


layout(location = 3) out Interpolants2
{
  flat uint vertexID;
}
OUTBARY[];
#endif
#endif

#ifndef MESHSHADER_WORKGROUP_SIZE


#define MESHSHADER_WORKGROUP_SIZE 32
#endif


layout(local_size_x = MESHSHADER_WORKGROUP_SIZE) in;


layout(max_vertices = CLUSTER_VERTEX_COUNT, max_primitives = CLUSTER_TRIANGLE_COUNT) out;


layout(triangles) out;

const uint MESHLET_VERTEX_ITERATIONS = ((CLUSTER_VERTEX_COUNT + MESHSHADER_WORKGROUP_SIZE - 1) / MESHSHADER_WORKGROUP_SIZE);
const uint MESHLET_TRIANGLE_ITERATIONS = ((CLUSTER_TRIANGLE_COUNT + MESHSHADER_WORKGROUP_SIZE - 1) / MESHSHADER_WORKGROUP_SIZE);

#if USE_TWO_SIDED


#define CULLING_NO_HIZ
#include "culling.glsl"
#endif

#if USE_EXT_MESH_SHADER && USE_TWO_SIDED
shared vec4 s_vertices[CLUSTER_VERTEX_COUNT];
#endif


// Emit one mesh-shader workgroup per visible cluster and fetch compact attributes from its resident payload.
void main()
{
#if USE_EXT_MESH_SHADER


uint workGroupID  = getWorkGroupIndexLinearized(gl_WorkGroupID);
  bool isValid      = workGroupID < build.numRenderedClusters;

  ClusterInfo cinfo = build.renderClusterInfos.d[min(workGroupID, build.numRenderedClusters-1)];
#else

  uint workGroupID  = gl_WorkGroupID.x;
  ClusterInfo cinfo = build.renderClusterInfos.d[workGroupID];
#endif

  uint instanceID = cinfo.instanceID;
  uint clusterID  = cinfo.clusterID;

  RenderInstance instance = instances[instanceID];
  Geometry geometry       = geometries[instance.geometryID];

#if USE_STREAMING

  Cluster_in clusterRef = Cluster_in(streaming.resident.clusters.d[clusterID]);
#else

  Cluster_in clusterRef = Cluster_in(geometry.preloadedClusters.d[clusterID]);
#endif
  Cluster cluster = clusterRef.d;
  uint vertMax = cluster.vertexCountMinusOne;
  uint triMax  = cluster.triangleCountMinusOne;

#if USE_EXT_MESH_SHADER

  uint vertCount = isValid ? vertMax + 1 : 0;
  uint triCount  = isValid ? triMax + 1 : 0;

  SetMeshOutputsEXT(vertCount, triCount);
  if (triCount == 0)
    return;
#else

  if (gl_LocalInvocationID.x == 0) {
    gl_PrimitiveCountNV = triMax + 1;
  }
#endif

#if USE_RENDER_STATS
  if (gl_LocalInvocationID.x == 0) {
    atomicAdd(readback.numRenderedTriangles, uint(triMax + 1));
  }
#endif

  vec3s_in  oVertices    = vec3s_in(Cluster_getVertexPositions(clusterRef));
  uint8s_in localIndices = uint8s_in(Cluster_getTriangleIndices(clusterRef));


  uint currentLodLevel = cluster.lodLevel;


  float lodTransitionFactor = 0.0;


  [[unroll]] for(uint i = 0; i < uint(MESHLET_VERTEX_ITERATIONS); i++)
  {

    uint vert        = gl_LocalInvocationID.x + i * MESHSHADER_WORKGROUP_SIZE;

    uint vertLoad    = min(vert, vertMax);

    vec3 oPos = oVertices.d[vertLoad];

    vec3 wPos = instance.worldMatrix * vec4(oPos, 1.0f);


    if(vert <= vertMax)
    {


      vec4 hPos = view.viewProjMatrix * vec4(wPos,1);

    #if USE_EXT_MESH_SHADER
      gl_MeshVerticesEXT[vert].gl_Position = hPos;
    #else
      gl_MeshVerticesNV[vert].gl_Position = hPos;
    #endif

    #if USE_EXT_MESH_SHADER && USE_TWO_SIDED
      s_vertices[vert] = hPos;
    #endif

    #if !USE_DEPTH_ONLY
    #if ALLOW_SHADING
      OUT[vert].wPos                      = wPos.xyz;
    #endif
    #if ALLOW_SHADING && (ALLOW_VERTEX_NORMALS || ALLOW_VERTEX_TEXCOORDS)
      OUTBARY[vert].vertexID              = vert;
    #endif
      OUT[vert].clusterID                 = clusterID;
      OUT[vert].instanceID                = instanceID;
    #endif
    }
  }

  [[unroll]] for(uint i = 0; i < uint(MESHLET_TRIANGLE_ITERATIONS); i++)
  {

    uint tri     = gl_LocalInvocationID.x + i * MESHSHADER_WORKGROUP_SIZE;

    uint triLoad = min(tri, triMax);

    uvec3 indices = uvec3(localIndices.d[triLoad * 3 + 0],
                          localIndices.d[triLoad * 3 + 1],
                          localIndices.d[triLoad * 3 + 2]);
#if !USE_FORCED_TWO_SIDED

    if (instance.flipWinding != 0
#if USE_TWO_SIDED && !USE_EXT_MESH_SHADER

      || (instance.twoSided != 0 && !isFrontFacingHW(gl_MeshVerticesNV[indices.x].gl_Position,
                                                     gl_MeshVerticesNV[indices.y].gl_Position,
                                                     gl_MeshVerticesNV[indices.z].gl_Position))
#elif USE_TWO_SIDED && USE_EXT_MESH_SHADER

      || (instance.twoSided != 0 && !isFrontFacingHW(s_vertices[indices.x],s_vertices[indices.y],s_vertices[indices.z]))
#endif
    )
    {

      indices.xy = indices.yx;
    }
#endif
    bool isRendered = tri <= triMax;
    uint triOut     = tri;

    if(isRendered)
    {


    #if USE_EXT_MESH_SHADER
      gl_PrimitiveTriangleIndicesEXT[triOut] = indices;
      #if !USE_DEPTH_ONLY

      gl_MeshPrimitivesEXT[triOut].gl_PrimitiveID = int(tri);
      #endif
    #else

      gl_PrimitiveIndicesNV[triOut * 3 + 0] = indices.x;
      gl_PrimitiveIndicesNV[triOut * 3 + 1] = indices.y;
      gl_PrimitiveIndicesNV[triOut * 3 + 2] = indices.z;
      #if !USE_DEPTH_ONLY

      gl_MeshPrimitivesNV[triOut].gl_PrimitiveID = int(tri);
      #endif
    #endif
    }
  }
}
