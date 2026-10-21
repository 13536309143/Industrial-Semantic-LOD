//==============================================================================
// src/meshlod/meshlod_build.h
// Main clustered LOD build loop.
// Each iteration clusters the current mesh, emits groups, simplifies toward the next level, and schedules optional parallel work.
//==============================================================================
#pragma once


#include "meshlod_impl.h"
#include "meshlod_bounds.h"
#include "meshlod_clustering.h"
#include "meshlod_simplify.h"
#include <chrono>

namespace clod
{

inline uint64_t timestampMicroseconds()
{
	return uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(
	                    std::chrono::steady_clock::now().time_since_epoch())
	                    .count());
}

inline void addMicroseconds(std::atomic_uint64_t& dst, uint64_t start)
{
	dst.fetch_add(timestampMicroseconds() - start, std::memory_order_relaxed);
}

size_t estimateSplitClusterCapacity(const clodConfig& config, const std::vector<Cluster>& clusters, const std::vector<std::vector<int>>& groups)
{
	size_t capacity = 0;
	const size_t minTriangles = std::max<size_t>(1, config.min_triangles);

	for (const std::vector<int>& group : groups)
	{
		size_t groupTriangles = 0;

		for (int clusterIndex : group)
		{
			assert(clusterIndex >= 0);
			groupTriangles += clusters[size_t(clusterIndex)].indices.size() / 3;
		}

		const size_t estimatedSplitCount = std::max<size_t>(1, (groupTriangles + minTriangles - 1) / minTriangles);
		capacity += estimatedSplitCount;
	}

	return capacity;
}


int outputGroup(const clodConfig& config, const clodMesh& mesh, const std::vector<Cluster>& clusters, const std::vector<int>& group, const clodBounds& simplified, int depth, void* output_context, clodOutput output_callback, size_t task_index, unsigned int thread_index)
{


	std::vector<clodCluster> group_clusters(group.size());

	for (size_t i = 0; i < group.size(); ++i)
	{
		const Cluster& cluster = clusters[group[i]];
		clodCluster& result = group_clusters[i];
		result.refined = cluster.refined;
		result.bounds = (config.optimize_bounds && cluster.refined != -1) ? boundsCompute(mesh, cluster.indices, cluster.bounds.error) : cluster.bounds;

		result.indices = cluster.indices.data();

		result.index_count = cluster.indices.size();
		result.vertex_count = cluster.vertices;
	}

	return output_callback ? output_callback(output_context, {depth, simplified}, group_clusters.data(), group_clusters.size(), task_index, thread_index) : -1;
}

}


clodConfig clodDefaultConfig(size_t max_triangles)
{

	assert(max_triangles >= 4 && max_triangles <= 256);
	clodConfig config = {};
	config.max_vertices = max_triangles;
	config.min_triangles = max_triangles / 3;
	config.max_triangles = max_triangles;
#if MESHOPTIMIZER_VERSION < 1000
	config.min_triangles &= ~3;
#endif
	config.partition_spatial = true;
	config.partition_size = 16;

	config.cluster_spatial = false;
	config.cluster_split_factor = 2.0f;

	config.optimize_clusters = true;
	config.simplify_ratio = 0.5f;
	config.simplify_threshold = 0.85f;

	config.simplify_error_merge_previous = 1.0f;
	config.simplify_error_factor_sloppy = 2.0f;
	config.simplify_permissive = true;
	config.simplify_fallback_permissive = true;
	config.simplify_fallback_sloppy = true;
	config.feature_constraints = true;
	config.feature_attribute_weight = 4.0f;
	config.feature_protect_threshold = 0.78f;
	config.feature_critical_threshold = 0.93f;
	config.semantic_priority = 0;
	config.semantic_confidence = 1.0f;
	config.feature_soft_scale = 1.0f;
	config.feature_hard_lock_ratio = 1.0f;
	config.hierarchy_depth_decay = 0.0f;
	config.hierarchy_min_ratio = config.simplify_ratio;
	config.simplify_error_relative = 0.0f;
	config.simplify_error_absolute = 0.0f;
	config.max_axis_shrink = 0.0f;
	config.preserve_components = false;
	config.base_simplify_ratio = 1.0f;
	config.base_error_relative = 0.0f;

	return config;
}


void clodBuild_iterationTask(void* iteration_context, void* output_context, size_t i, unsigned int thread_index)
{
	using namespace clod;

	IterationContext& context = *(IterationContext*)iteration_context;
	const std::vector<std::vector<int>>& groups = context.groups;
	std::vector<Cluster>& clusters = context.clusters;
	const std::vector<unsigned char>& locks = context.locks;
	const clodMesh& mesh = context.mesh;
	const clodConfig& config = context.config;
	int                                  depth = context.depth;

	const uint64_t featureStart = mesh.timing_stats ? timestampMicroseconds() : 0;

	std::vector<unsigned int> merged;
	merged.reserve(groups[i].size() * config.max_triangles * 3);

	for (size_t j = 0; j < groups[i].size(); ++j)
		merged.insert(merged.end(), clusters[groups[i][j]].indices.begin(), clusters[groups[i][j]].indices.end());

	float depth_ratio = config.simplify_ratio - std::max(0, depth - 1) * config.hierarchy_depth_decay;
	depth_ratio = std::max(config.hierarchy_min_ratio, std::min(config.simplify_ratio, depth_ratio));
	size_t target_size = size_t((merged.size() / 3) * depth_ratio) * 3;


	clodBounds bounds = boundsMerge(clusters, groups[i]);

	float error = 0.f;


	std::vector<unsigned int> simplified = simplify(config, mesh, merged, locks, target_size, &error);

	if (mesh.timing_stats)
		addMicroseconds(mesh.timing_stats->feature_lod_microseconds, featureStart);

	if (simplified.size() > merged.size() * config.simplify_threshold)
	{
		bounds.error = FLT_MAX;
		outputGroup(config, mesh, clusters, groups[i], bounds, depth, output_context, context.output_callback, i, thread_index);
		return;
	}

	bounds.error = std::max(bounds.error * config.simplify_error_merge_previous, error) + error * config.simplify_error_merge_additive;


	int refined = outputGroup(config, mesh, clusters, groups[i], bounds, depth, output_context, context.output_callback, i, thread_index);

	for (size_t j = 0; j < groups[i].size(); ++j)
		clusters[groups[i][j]].indices = std::vector<unsigned int>();

	const uint64_t clusterStart = mesh.timing_stats ? timestampMicroseconds() : 0;
	std::vector<Cluster> split = clusterize(config, mesh, simplified.data(), simplified.size());
	if (mesh.timing_stats)
		addMicroseconds(mesh.timing_stats->cluster_partition_microseconds, clusterStart);

	size_t cluster_index = context.next_cluster.fetch_add(split.size());
	size_t pending_index = context.next_pending.fetch_add(split.size());

	for (Cluster& cluster : split)
	{
		cluster.refined = refined;

		cluster.bounds = bounds;

		assert(pending_index < context.pending.size());
		assert(cluster_index < context.clusters.size());


		context.pending[pending_index++] = int(cluster_index);

		context.clusters[cluster_index++] = std::move(cluster);
	}
}


size_t clodBuild(clodConfig config, clodMesh mesh, void* output_context, clodOutput output_callback, clodIteration iteration_callback)
{
	using namespace clod;

	assert(mesh.vertex_attributes_stride % sizeof(float) == 0);
	assert(mesh.attribute_count * sizeof(float) <= mesh.vertex_attributes_stride);
	assert(mesh.attribute_protect_mask < (1u << (mesh.vertex_attributes_stride / sizeof(float))));

	IterationContext context;
	context.config = config;
	context.mesh = mesh;
	context.output_callback = output_callback;

	context.locks.resize(mesh.vertex_count);

	context.remap.resize(mesh.vertex_count);


	meshopt_generatePositionRemap(&context.remap[0], mesh.vertex_positions, mesh.vertex_count, mesh.vertex_positions_stride);
	context.mesh.component_ids = nullptr;
	if(config.simplify_error_relative > 0)
		context.config.simplify_error_absolute = config.simplify_error_relative * meshopt_simplifyScale(mesh.vertex_positions, mesh.vertex_count, mesh.vertex_positions_stride);
	if(config.preserve_components)
	{
		// Weld seam duplicates before unioning faces. Storage is transient during build.
		context.component_ids = context.remap;
		auto root = [&](unsigned int v) {
			while(context.component_ids[v] != v)
			{
				context.component_ids[v] = context.component_ids[context.component_ids[v]];
				v = context.component_ids[v];
			}
			return v;
		};
		for(size_t i = 0; i < mesh.index_count; i += 3)
		{
			unsigned int a = root(mesh.indices[i]);
			for(size_t j = 1; j < 3; ++j)
			{
				unsigned int b = root(mesh.indices[i+j]);
				context.component_ids[b] = a;
			}
		}
		unsigned int first = mesh.index_count ? root(mesh.indices[0]) : 0;
		bool multiple = false;
		for(size_t i = 0; i < mesh.vertex_count; ++i)
			context.component_ids[i] = root(unsigned(i));
		for(size_t i = 0; i < mesh.index_count; ++i)
			multiple |= context.component_ids[mesh.indices[i]] != first;
		if(multiple)
			context.mesh.component_ids = context.component_ids.data();
	}
	if (mesh.attribute_protect_mask)
	{
		size_t max_attributes = mesh.vertex_attributes_stride / sizeof(float);

		for (size_t i = 0; i < mesh.vertex_count; ++i)
		{
			unsigned int r = context.remap[i];
			for (size_t j = 0; j < max_attributes; ++j)
				if (r != i && (mesh.attribute_protect_mask & (1u << j)) && mesh.vertex_attributes[i * max_attributes + j] != mesh.vertex_attributes[r * max_attributes + j])
					context.locks[i] |= meshopt_SimplifyVertex_Protect;
		}
	}

	if (mesh.timing_stats)
	{
		const uint64_t featureStart = timestampMicroseconds();
		analyzeFeatureConstraints(context);
		addMicroseconds(mesh.timing_stats->feature_lod_microseconds, featureStart);
	}
	else
	{
		analyzeFeatureConstraints(context);
	}

	float baseError = 0.0f;
	if(config.base_simplify_ratio < 1.0f && config.base_error_relative > 0.0f)
	{
		// Reduce redundant source tessellation before creating the finest clusters.
		// Positions/attributes keep their original indexing; no instance is removed.
		clodConfig baseConfig = context.config;
		baseConfig.simplify_error_absolute = config.base_error_relative * meshopt_simplifyScale(mesh.vertex_positions, mesh.vertex_count, mesh.vertex_positions_stride);
		baseConfig.simplify_fallback_permissive = false;
		baseConfig.simplify_fallback_sloppy = false;
		baseConfig.max_axis_shrink = 0.005f;
		std::vector<unsigned int> source(mesh.indices, mesh.indices + mesh.index_count);
		context.base_indices = simplify(baseConfig, context.mesh, source, context.locks,
		    std::max(size_t(3), size_t(float(mesh.index_count / 3) * config.base_simplify_ratio) * 3), &baseError);
		context.mesh.indices = context.base_indices.data();
		context.mesh.index_count = context.base_indices.size();
	}
	uint64_t clusterStart = mesh.timing_stats ? timestampMicroseconds() : 0;
	context.clusters = clusterize(config, context.mesh, context.mesh.indices, context.mesh.index_count);
	if (mesh.timing_stats)
		addMicroseconds(mesh.timing_stats->cluster_partition_microseconds, clusterStart);

	context.next_cluster = context.clusters.size();

	for (Cluster& cluster : context.clusters)

		cluster.bounds = boundsCompute(context.mesh, cluster.indices, baseError);

	context.pending.resize(context.clusters.size());
	for (size_t i = 0; i < context.clusters.size(); ++i)

		context.pending[i] = int(i);

	while (context.pending.size() > 1)
	{

		clusterStart = mesh.timing_stats ? timestampMicroseconds() : 0;
		context.groups = partition(config, mesh, context.clusters, context.pending, context.remap);
		if (mesh.timing_stats)
			addMicroseconds(mesh.timing_stats->cluster_partition_microseconds, clusterStart);

		const size_t splitCapacity = estimateSplitClusterCapacity(config, context.clusters, context.groups);

		context.clusters.resize(context.clusters.size() + splitCapacity);
		context.pending.resize(splitCapacity);
		context.next_pending = 0;


		if (mesh.timing_stats)
		{
			const uint64_t featureStart = timestampMicroseconds();
			lockBoundary(context.locks, context.groups, context.clusters, context.remap, context.mesh.vertex_lock);
			addMicroseconds(mesh.timing_stats->feature_lod_microseconds, featureStart);
		}
		else
		{
			lockBoundary(context.locks, context.groups, context.clusters, context.remap, context.mesh.vertex_lock);
		}

		if (iteration_callback)
		{
			iteration_callback(&context, output_context, context.depth, context.groups.size());
		}
		else
		{
			for (size_t i = 0; i < context.groups.size(); ++i)
			{

				clodBuild_iterationTask(&context, output_context, i, 0);
			}
		}


		context.pending.resize(context.next_pending);

		context.clusters.resize(context.next_cluster);

		context.depth++;
	}

	if (context.pending.size())
	{


		clodBounds bounds = boundsMerge(context.clusters, context.pending);
		bounds.error = FLT_MAX;

		outputGroup(config, mesh, context.clusters, context.pending, bounds, context.depth, output_context, output_callback, 0, 0);
	}

	return context.clusters.size();
}
