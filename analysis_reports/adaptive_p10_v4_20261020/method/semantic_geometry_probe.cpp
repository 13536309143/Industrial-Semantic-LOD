#include <meshoptimizer.h>
#include <cstddef>
#include <cstdint>
#include "meshlod.h"
#include "meshlod_impl.h"

#if defined(_WIN32)
#define PROBE_EXPORT extern "C" __declspec(dllexport)
#else
#define PROBE_EXPORT extern "C" __attribute__((visibility("default")))
#endif

// A single-mesh QEM diagnostic, independent of the application's cluster hierarchy.
PROBE_EXPORT size_t semantic_probe_simplify(uint32_t* output, const uint32_t* indices, size_t indexCount,
                                           const float* positions, size_t vertexCount, size_t targetCount,
                                           float* relativeError)
{
  if(!output || !indices || !positions || !relativeError || indexCount % 3 || vertexCount < 3)
    return 0;
  targetCount = targetCount / 3 * 3;
  return meshopt_simplify(output, indices, indexCount, positions, vertexCount, sizeof(float) * 3,
                         targetCount, 1.0f, 0, relativeError);
}

// Exercise the same bounded simplifier and guards used by the cluster builder.
// This still evaluates one mesh, not an entire rendered hierarchy.
PROBE_EXPORT size_t semantic_probe_simplify_v4(uint32_t* output, const uint32_t* indices, size_t indexCount,
    const float* positions, size_t vertexCount, size_t targetCount, const uint32_t* componentIds,
    float errorRelative, float axisShrink, float* absoluteError)
{
  if(!output || !indices || !positions || !absoluteError || indexCount % 3 || vertexCount < 3)
    return 0;
  clodConfig config = clodDefaultConfig(128);
  config.simplify_error_absolute = errorRelative * meshopt_simplifyScale(positions, vertexCount, sizeof(float)*3);
  config.simplify_fallback_sloppy = false;
  config.simplify_fallback_permissive = false;
  config.simplify_regularize = false;
  config.simplify_error_edge_limit = 0;
  config.max_axis_shrink = axisShrink;
  config.preserve_components = true;
  clodMesh mesh = {};
  mesh.vertex_positions = positions; mesh.vertex_count = vertexCount; mesh.vertex_positions_stride = sizeof(float)*3;
  mesh.component_ids = componentIds;
  std::vector<unsigned int> source(indices, indices+indexCount);
  std::vector<unsigned char> locks(vertexCount, 0);
  auto lod = clod::simplify(config, mesh, source, locks, targetCount/3*3, absoluteError);
  std::copy(lod.begin(), lod.end(), output);
  return lod.size();
}

struct ProbeHierarchy { uint64_t groups=0, clusters=0, indices=0, vertices=0, depth=0, finestIndices=0; };
static int probeGroup(void* data, clodGroup group, const clodCluster* clusters, size_t count, size_t, unsigned int)
{
  auto& r = *static_cast<ProbeHierarchy*>(data);
  r.groups++; r.clusters+=count; r.depth=std::max(r.depth,uint64_t(group.depth));
  for(size_t i=0;i<count;++i) { r.indices+=clusters[i].index_count; r.vertices+=clusters[i].vertex_count; if(group.depth==0)r.finestIndices+=clusters[i].index_count; }
  return int(r.groups-1);
}
PROBE_EXPORT int semantic_probe_hierarchy(const uint32_t* indices, size_t indexCount, const float* positions,
    size_t vertexCount, float nearRatio, float farRatio, float lockBudget, float errorRelative,
    float axisShrink, uint64_t* counters)
{
  if(!indices || !positions || !counters || indexCount%3 || vertexCount<3) return 0;
  clodConfig config=clodDefaultConfig(128);
  config.simplify_ratio=nearRatio; config.hierarchy_min_ratio=farRatio; config.hierarchy_depth_decay=.08f;
  config.feature_hard_lock_ratio=lockBudget; config.semantic_priority=5;
  config.feature_soft_scale=.85f; config.feature_attribute_weight=.8f;
  config.simplify_error_relative=errorRelative; config.max_axis_shrink=axisShrink;
  config.preserve_components=errorRelative>0; config.simplify_fallback_sloppy=false;
  clodMesh mesh={}; mesh.indices=indices; mesh.index_count=indexCount; mesh.vertex_count=vertexCount;
  mesh.vertex_positions=positions; mesh.vertex_positions_stride=sizeof(float)*3;
  ProbeHierarchy result;
  clodBuild(config,mesh,&result,probeGroup,nullptr);
  counters[0]=result.groups; counters[1]=result.clusters; counters[2]=result.indices;
  counters[3]=result.vertices; counters[4]=result.depth;
  return 1;
}

PROBE_EXPORT int semantic_probe_base_hierarchy(const uint32_t* indices, size_t indexCount, const float* positions,
    size_t vertexCount, float baseRatio, float baseError, uint64_t* counters)
{
  if(!indices || !positions || !counters || indexCount%3 || vertexCount<3) return 0;
  clodConfig config=clodDefaultConfig(128);
  config.base_simplify_ratio=baseRatio; config.base_error_relative=baseError;
  config.feature_hard_lock_ratio=.035f; config.feature_soft_scale=.85f; config.semantic_priority=5;
  config.preserve_components=true; config.simplify_fallback_sloppy=false;
  config.simplify_error_relative=.01f; config.max_axis_shrink=.025f;
  clodMesh mesh={};mesh.indices=indices;mesh.index_count=indexCount;mesh.vertex_count=vertexCount;
  mesh.vertex_positions=positions;mesh.vertex_positions_stride=sizeof(float)*3;
  ProbeHierarchy result;clodBuild(config,mesh,&result,probeGroup,nullptr);
  counters[0]=result.groups;counters[1]=result.clusters;counters[2]=result.indices;
  counters[3]=result.vertices;counters[4]=result.depth;
  counters[5]=result.finestIndices;
  return 1;
}
