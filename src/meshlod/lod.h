//==============================================================================
// src/meshlod/lod.h
// Public C-style declarations for clustered LOD generation.
// The types describe source mesh attributes, simplification settings, cluster bounds, and callback-based output.
//==============================================================================
#pragma once


#include <stddef.h>


struct clodConfig
{
	size_t max_vertices;
	size_t min_triangles;
	size_t max_triangles;
	bool partition_spatial;
	bool partition_sort;
	size_t partition_size;
	bool cluster_spatial;
	float cluster_fill_weight;
	float cluster_split_factor;
	float simplify_ratio;
	float simplify_threshold;
	float simplify_error_merge_previous;
	float simplify_error_merge_additive;
	float simplify_error_factor_sloppy;
	float simplify_error_edge_limit;
	bool simplify_permissive;
	bool simplify_fallback_permissive;
	bool simplify_fallback_sloppy;
	bool simplify_regularize;
	bool optimize_bounds;
	bool optimize_clusters;
};


struct clodMesh
{
	const unsigned int* indices;
	size_t index_count;
	size_t vertex_count;
	const float* vertex_positions;
	size_t vertex_positions_stride;
	const float* vertex_attributes;
	size_t vertex_attributes_stride;
	const unsigned char* vertex_lock;
	const float* attribute_weights;
	size_t attribute_count;
	unsigned int attribute_protect_mask;
};


struct clodBounds
{
	float center[3];
	float radius;
	float error;
};


struct clodCluster
{
	int refined;
	clodBounds bounds;
	const unsigned int* indices;
	size_t index_count;
	size_t vertex_count;
};


struct clodGroup
{
	int depth;
	clodBounds simplified;
};


typedef int (*clodOutput)(void* output_context, clodGroup group, const clodCluster* clusters, size_t cluster_count, size_t task_index, unsigned int thread_index);


typedef void (*clodIteration)(void* iteration_context, void* output_context, int depth, size_t task_count);
#ifdef __cplusplus
extern "C"
{
#endif


clodConfig clodDefaultConfig(size_t max_triangles);


size_t clodBuild(clodConfig config, clodMesh mesh, void* output_context, clodOutput output_callback, clodIteration iteration_callback);


void clodBuild_iterationTask(void* iteration_context, void* output_context, size_t task_index, unsigned int thread_index);


size_t clodLocalIndices(unsigned int* vertices, unsigned char* triangles, const unsigned int* indices, size_t index_count);
#ifdef __cplusplus
}
template <typename Output>


size_t clodBuild(clodConfig config, clodMesh mesh, Output output)
{


	struct Call
	{


		static int output(void* output_context, clodGroup group, const clodCluster* clusters, size_t cluster_count)
		{
			return (*static_cast<Output*>(output_context))(group, clusters, cluster_count);
		}
	};

	return clodBuild(config, mesh, &output, &Call::output, nullptr);
}
#endif
#ifdef CLUSTERLOD

#include <float.h>
#include <math.h>
#include <string.h>
#include <algorithm>
#include <vector>
#include <atomic>

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


static clodBounds boundsCompute(const clodMesh& mesh, const std::vector<unsigned int>& indices, float error)
{
	meshopt_Bounds bounds = meshopt_computeClusterBounds(&indices[0], indices.size(), mesh.vertex_positions, mesh.vertex_count, mesh.vertex_positions_stride);
	clodBounds result;
	result.center[0] = bounds.center[0];
	result.center[1] = bounds.center[1];
	result.center[2] = bounds.center[2];
	result.radius = bounds.radius;
	result.error = error;
	return result;
}


static clodBounds boundsMerge(const std::vector<Cluster>& clusters, const std::vector<int>& group)
{

	clodBounds result = {};


	if (group.size() == 1) {
		result = clusters[group[0]].bounds;
		return result;
	}


	float centers[64 * 3];
	float radii[64];
	size_t count = std::min(group.size(), size_t(64));

	for (size_t j = 0; j < count; ++j) {
		const clodBounds& b = clusters[group[j]].bounds;
		centers[j * 3 + 0] = b.center[0];
		centers[j * 3 + 1] = b.center[1];
		centers[j * 3 + 2] = b.center[2];
		radii[j] = b.radius;

		result.error = std::max(result.error, b.error);
	}

	meshopt_Bounds merged = meshopt_computeSphereBounds(centers, count, sizeof(float) * 3, radii, sizeof(float));
	result.center[0] = merged.center[0];
	result.center[1] = merged.center[1];
	result.center[2] = merged.center[2];
	result.radius = merged.radius;

	return result;
}


static std::vector<Cluster> clusterize(const clodConfig& config, const clodMesh& mesh, const unsigned int* indices, size_t index_count)
{


	size_t max_meshlets = meshopt_buildMeshletsBound(index_count, config.max_vertices, config.min_triangles);


	std::vector<meshopt_Meshlet> meshlets(max_meshlets);


	std::vector<unsigned int> meshlet_vertices(index_count);

#if MESHOPTIMIZER_VERSION < 1000


	std::vector<unsigned char> meshlet_triangles(index_count + max_meshlets * 3);
#else


	std::vector<unsigned char> meshlet_triangles(index_count);
#endif

	if (config.cluster_spatial)

		meshlets.resize(meshopt_buildMeshletsSpatial(meshlets.data(), meshlet_vertices.data(), meshlet_triangles.data(), indices, index_count,
		    mesh.vertex_positions, mesh.vertex_count, mesh.vertex_positions_stride,
		    config.max_vertices, config.min_triangles, config.max_triangles, config.cluster_fill_weight));
	else

		meshlets.resize(meshopt_buildMeshletsFlex(meshlets.data(), meshlet_vertices.data(), meshlet_triangles.data(), indices, index_count,
		    mesh.vertex_positions, mesh.vertex_count, mesh.vertex_positions_stride,
		    config.max_vertices, config.min_triangles, config.max_triangles, 0.f, config.cluster_split_factor));


	std::vector<Cluster> clusters;
	clusters.reserve(meshlets.size());

	for (size_t i = 0; i < meshlets.size(); ++i)
	{
		const meshopt_Meshlet& meshlet = meshlets[i];
		Cluster cluster;

		if (config.optimize_clusters)

			meshopt_optimizeMeshlet(&meshlet_vertices[meshlet.vertex_offset], &meshlet_triangles[meshlet.triangle_offset], meshlet.triangle_count, meshlet.vertex_count);

		cluster.vertices = meshlet.vertex_count;


		cluster.indices.resize(meshlet.triangle_count * 3);

		for (size_t j = 0; j < meshlet.triangle_count * 3; ++j)
			cluster.indices[j] = meshlet_vertices[meshlet.vertex_offset + meshlet_triangles[meshlet.triangle_offset + j]];

		cluster.group = -1;
		cluster.refined = -1;
		clusters.push_back(std::move(cluster));
	}

	return clusters;
}


static std::vector<std::vector<int> > partition(const clodConfig& config, const clodMesh& mesh, const std::vector<Cluster>& clusters, const std::vector<int>& pending, const std::vector<unsigned int>& remap)
{

	if (pending.size() <= config.partition_size)
		return {pending};

	std::vector<unsigned int> cluster_indices;


	std::vector<unsigned int> cluster_counts(pending.size());
	size_t total_index_count = 0;
	for (size_t i = 0; i < pending.size(); ++i)

		total_index_count += clusters[pending[i]].indices.size();


	cluster_indices.reserve(total_index_count);

	for (size_t i = 0; i < pending.size(); ++i)
	{
		const Cluster& cluster = clusters[pending[i]];
		cluster_counts[i] = unsigned(cluster.indices.size());

		for (size_t j = 0; j < cluster.indices.size(); ++j)

			cluster_indices.push_back(remap[cluster.indices[j]]);
	}


	std::vector<unsigned int> cluster_part(pending.size());

	size_t partition_count = meshopt_partitionClusters(&cluster_part[0], &cluster_indices[0], cluster_indices.size(), &cluster_counts[0], cluster_counts.size(),
	    config.partition_spatial ? mesh.vertex_positions : NULL, remap.size(), mesh.vertex_positions_stride, config.partition_size);


	std::vector<std::vector<int> > partitions(partition_count);
	for (size_t i = 0; i < partition_count; ++i)

		partitions[i].reserve(config.partition_size + config.partition_size / 3);

	std::vector<unsigned int> partition_remap;

	if (config.partition_sort)
	{


		std::vector<float> partition_point(partition_count * 3);
		for (size_t i = 0; i < pending.size(); ++i)
			memcpy(&partition_point[cluster_part[i] * 3], clusters[pending[i]].bounds.center, sizeof(float) * 3);

		partition_remap.resize(partition_count);

		meshopt_spatialSortRemap(partition_remap.data(), partition_point.data(), partition_count, sizeof(float) * 3);
	}

	for (size_t i = 0; i < pending.size(); ++i)

		partitions[partition_remap.empty() ? cluster_part[i] : partition_remap[cluster_part[i]]].push_back(pending[i]);

	return partitions;
}


static void lockBoundary(std::vector<unsigned char>& locks, const std::vector<std::vector<int> >& groups, const std::vector<Cluster>& clusters, const std::vector<unsigned int>& remap, const unsigned char* vertex_lock)
{

	for (size_t i = 0; i < locks.size(); ++i)
		locks[i] &= ~((1 << 0) | (1 << 7));

	for (size_t i = 0; i < groups.size(); ++i)
	{

		for (size_t j = 0; j < groups[i].size(); ++j)
		{
			const Cluster& cluster = clusters[groups[i][j]];

			for (size_t k = 0; k < cluster.indices.size(); ++k)
			{
				unsigned int v = cluster.indices[k];
				unsigned int r = remap[v];

				locks[r] |= locks[r] >> 7;
			}
		}

		for (size_t j = 0; j < groups[i].size(); ++j)
		{
			const Cluster& cluster = clusters[groups[i][j]];

			for (size_t k = 0; k < cluster.indices.size(); ++k)
			{
				unsigned int v = cluster.indices[k];
				unsigned int r = remap[v];

				locks[r] |= 1 << 7;
			}
		}
	}

	for (size_t i = 0; i < locks.size(); ++i)
	{
		unsigned int r = remap[i];

		locks[i] = (locks[r] & 1) | (locks[i] & meshopt_SimplifyVertex_Protect);

		if (vertex_lock)
			locks[i] |= vertex_lock[i];
	}
}


struct SloppyVertex
{
	float x, y, z;
	unsigned int id;
};


static void simplifyFallback(std::vector<unsigned int>& lod, const clodMesh& mesh, const std::vector<unsigned int>& indices, const std::vector<unsigned char>& locks, size_t target_count, float* error)
{


	std::vector<SloppyVertex> subset(indices.size());


	std::vector<unsigned char> subset_locks(indices.size());
	lod.resize(indices.size());

	size_t positions_stride = mesh.vertex_positions_stride / sizeof(float);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		unsigned int v = indices[i];

		assert(v < mesh.vertex_count);

		subset[i].x = mesh.vertex_positions[v * positions_stride + 0];
		subset[i].y = mesh.vertex_positions[v * positions_stride + 1];
		subset[i].z = mesh.vertex_positions[v * positions_stride + 2];
		subset[i].id = v;

		subset_locks[i] = locks[v];

		lod[i] = unsigned(i);
	}


	lod.resize(meshopt_simplifySloppy(&lod[0], &lod[0], lod.size(), &subset[0].x, subset.size(), sizeof(SloppyVertex), subset_locks.data(), target_count, FLT_MAX, error));

	*error *= meshopt_simplifyScale(&subset[0].x, subset.size(), sizeof(SloppyVertex));

	for (size_t i = 0; i < lod.size(); ++i)
		lod[i] = subset[lod[i]].id;
}


static std::vector<unsigned int> simplify(const clodConfig& config, const clodMesh& mesh, const std::vector<unsigned int>& indices, const std::vector<unsigned char>& locks, size_t target_count, float* error)
{

	if (target_count > indices.size())
		return indices;


	std::vector<unsigned int> lod(indices.size());

	unsigned int options = meshopt_SimplifySparse | meshopt_SimplifyErrorAbsolute | (config.simplify_permissive ? meshopt_SimplifyPermissive : 0) | (config.simplify_regularize ? meshopt_SimplifyRegularize : 0);

	lod.resize(meshopt_simplifyWithAttributes(&lod[0], &indices[0], indices.size(),
	    mesh.vertex_positions, mesh.vertex_count, mesh.vertex_positions_stride,
	    mesh.vertex_attributes, mesh.vertex_attributes_stride, mesh.attribute_weights, mesh.attribute_count,
	    &locks[0], target_count, FLT_MAX, options, error));


	if (lod.size() > target_count && config.simplify_fallback_permissive && !config.simplify_permissive)
		lod.resize(meshopt_simplifyWithAttributes(&lod[0], &indices[0], indices.size(),
		    mesh.vertex_positions, mesh.vertex_count, mesh.vertex_positions_stride,
		    mesh.vertex_attributes, mesh.vertex_attributes_stride, mesh.attribute_weights, mesh.attribute_count,
		    &locks[0], target_count, FLT_MAX, options | meshopt_SimplifyPermissive, error));

	if (lod.size() > target_count && config.simplify_fallback_sloppy)
	{

		simplifyFallback(lod, mesh, indices, locks, target_count, error);

		*error *= config.simplify_error_factor_sloppy;
	}

	if (config.simplify_error_edge_limit > 0)
	{
		float max_edge_sq = 0;

		for (size_t i = 0; i < indices.size(); i += 3)
		{
			unsigned int a = indices[i + 0], b = indices[i + 1], c = indices[i + 2];

			assert(a < mesh.vertex_count && b < mesh.vertex_count && c < mesh.vertex_count);

			const float* va = &mesh.vertex_positions[a * (mesh.vertex_positions_stride / sizeof(float))];
			const float* vb = &mesh.vertex_positions[b * (mesh.vertex_positions_stride / sizeof(float))];
			const float* vc = &mesh.vertex_positions[c * (mesh.vertex_positions_stride / sizeof(float))];
			float eab = (va[0] - vb[0]) * (va[0] - vb[0]) + (va[1] - vb[1]) * (va[1] - vb[1]) + (va[2] - vb[2]) * (va[2] - vb[2]);
			float eac = (va[0] - vc[0]) * (va[0] - vc[0]) + (va[1] - vc[1]) * (va[1] - vc[1]) + (va[2] - vc[2]) * (va[2] - vc[2]);
			float ebc = (vb[0] - vc[0]) * (vb[0] - vc[0]) + (vb[1] - vc[1]) * (vb[1] - vc[1]) + (vb[2] - vc[2]) * (vb[2] - vc[2]);

			float emax = std::max(std::max(eab, eac), ebc);
			float emin = std::min(std::min(eab, eac), ebc);
			max_edge_sq = std::max(max_edge_sq, std::max(emin, emax / 4));
		}

		*error = std::min(*error, sqrtf(max_edge_sq) * config.simplify_error_edge_limit);
	}

	return lod;
}


static int outputGroup(const clodConfig& config, const clodMesh& mesh, const std::vector<Cluster>& clusters, const std::vector<int>& group, const clodBounds& simplified, int depth, void* output_context, clodOutput output_callback, size_t task_index, unsigned int thread_index)
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


struct IterationContext
{
	clodConfig config;
	clodMesh   mesh;
	clodOutput output_callback = nullptr;
	std::vector<unsigned char> locks;
	std::vector<unsigned int>  remap;

	int depth = 0;
	std::vector<Cluster> clusters;
	std::atomic<size_t>  next_cluster = {};
	std::vector<std::vector<int>> groups;

	std::vector<int>    pending;
	std::atomic<size_t> next_pending = {};
};
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


	std::vector<unsigned int> merged;

	merged.reserve(groups[i].size() * config.max_triangles * 3);


	for (size_t j = 0; j < groups[i].size(); ++j)
		merged.insert(merged.end(), clusters[groups[i][j]].indices.begin(), clusters[groups[i][j]].indices.end());


	size_t target_size = size_t((merged.size() / 3) * config.simplify_ratio) * 3;


	clodBounds bounds = boundsMerge(clusters, groups[i]);

	float error = 0.f;


	std::vector<unsigned int> simplified = simplify(config, mesh, merged, locks, target_size, &error);


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


	std::vector<Cluster> split = clusterize(config, mesh, simplified.data(), simplified.size());


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


	context.clusters = clusterize(config, mesh, mesh.indices, mesh.index_count);

	context.next_cluster = context.clusters.size();


	for (Cluster& cluster : context.clusters)

		cluster.bounds = boundsCompute(mesh, cluster.indices, 0.f);


	context.pending.resize(context.clusters.size());
	for (size_t i = 0; i < context.clusters.size(); ++i)

		context.pending[i] = int(i);


	while (context.pending.size() > 1)
	{


		context.groups = partition(config, mesh, context.clusters, context.pending, context.remap);


		context.clusters.resize(context.clusters.size() + context.pending.size() + context.groups.size());
		context.pending.resize(context.pending.size() + context.groups.size());
		context.next_pending = 0;


		lockBoundary(context.locks, context.groups, context.clusters, context.remap, context.mesh.vertex_lock);


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


size_t clodLocalIndices(unsigned int* vertices, unsigned char* triangles, const unsigned int* indices, size_t index_count)
{
	size_t unique = 0;


	static constexpr size_t CACHE_SIZE = 4096;


	struct CacheEntry {
		unsigned int vertex_id;
		unsigned short local_index;
		CacheEntry* next;
	};


	static CacheEntry cachePool[CACHE_SIZE * 2];
	static CacheEntry* cache[CACHE_SIZE];
	static bool cacheInitialized = false;

	if (!cacheInitialized) {

		memset(cache, 0, sizeof(cache));
		cacheInitialized = true;
	}


	for (size_t i = 0; i < index_count; ++i)
	{
		unsigned int v = indices[i];


		unsigned int key = (v * 2654435761u) & (CACHE_SIZE - 1);


		CacheEntry* entry = cache[key];
		while (entry != nullptr) {
			if (entry->vertex_id == v) {
				triangles[i] = (unsigned char)entry->local_index;
				goto found;
			}
			entry = entry->next;
		}


		entry = &cachePool[unique];
		entry->vertex_id = v;
		entry->local_index = (unsigned short)unique;
		entry->next = cache[key];
		cache[key] = entry;

		triangles[i] = (unsigned char)unique;
		vertices[unique++] = v;

found:
		continue;
	}


	assert(unique <= 256);
	return unique;
}
#endif
