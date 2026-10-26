//==============================================================================
// src/meshlod/meshlod_impl.h
// Internal declarations for the clustered LOD builder.
// Implementation files share these transient cluster, vertex, and iteration structures while the public API stays compact.
//==============================================================================
#pragma once


#include "meshlod_types.h"
#include <float.h>
#include <math.h>
#include <string.h>
#include <cassert>
#include <algorithm>
#include <vector>
#include <atomic>
#include <unordered_map>
#include <queue>
#include <array>
#include <meshoptimizer.h>

namespace clod
{


struct Cluster
{
	size_t vertices;
	std::vector<unsigned int> indices;
	int group;
	int refined;
	clodBounds bounds;
};


struct SloppyVertex
{
	float x, y, z;
	unsigned int id;
};


struct IterationContext
{
	clodConfig config;
	clodMesh   mesh;
	clodOutput output_callback = nullptr;
	std::vector<unsigned char> locks;
	std::vector<unsigned int>  remap;
	std::vector<unsigned int> component_ids;
	std::vector<unsigned int> base_indices;
	std::vector<float> feature_importance;
	std::vector<float> feature_normals;
	std::vector<unsigned char> feature_locks;
    std::vector<unsigned char> feature_seam_protect;
	std::vector<float> feature_attributes;
	std::vector<float> feature_attribute_weights;

	int depth = 0;
	std::vector<Cluster> clusters;
	std::atomic<size_t>  next_cluster = {};
	std::vector<std::vector<int>> groups;

	std::vector<int>    pending;
	std::atomic<size_t> next_pending = {};
};


clodBounds boundsCompute(const clodMesh& mesh, const std::vector<unsigned int>& indices, float error);


clodBounds boundsMerge(const std::vector<Cluster>& clusters, const std::vector<int>& group);


std::vector<Cluster> clusterize(const clodConfig& config, const clodMesh& mesh, const unsigned int* indices, size_t index_count);


std::vector<std::vector<int> > partition(const clodConfig& config, const clodMesh& mesh, const std::vector<Cluster>& clusters, const std::vector<int>& pending, const std::vector<unsigned int>& remap);


void lockBoundary(std::vector<unsigned char>& locks, const std::vector<std::vector<int> >& groups, const std::vector<Cluster>& clusters, const std::vector<unsigned int>& remap, const unsigned char* vertex_lock);


void simplifyFallback(std::vector<unsigned int>& lod, const clodMesh& mesh, const std::vector<unsigned int>& indices, const std::vector<unsigned char>& locks, size_t target_count, float* error);


std::vector<unsigned int> simplify(const clodConfig& config, const clodMesh& mesh, const std::vector<unsigned int>& indices, const std::vector<unsigned char>& locks, size_t target_count, float* error);


int outputGroup(const clodConfig& config, const clodMesh& mesh, const std::vector<Cluster>& clusters, const std::vector<int>& group, const clodBounds& simplified, int depth, void* output_context, clodOutput output_callback, size_t task_index, unsigned int thread_index);

}
