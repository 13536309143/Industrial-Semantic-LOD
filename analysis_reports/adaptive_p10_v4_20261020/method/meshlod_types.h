//==============================================================================
// src/meshlod/meshlod_types.h
// Shared clustered LOD data types and callback signatures.
// These structures are the CPU-side contract between mesh preprocessing, simplification, grouping, and scene packing.
//==============================================================================
#pragma once


#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#include <atomic>

struct clodTimingStats
{
	std::atomic_uint64_t cluster_partition_microseconds = 0;
	std::atomic_uint64_t feature_lod_microseconds = 0;
};
#endif


// Tuning knobs for cluster size, simplification, feature protection, and output scheduling.
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
	bool feature_constraints;
	float feature_attribute_weight;
	float feature_protect_threshold;
	float feature_critical_threshold;
	int semantic_priority;
	float semantic_confidence;
	float feature_soft_scale;
	float feature_hard_lock_ratio;
	float hierarchy_depth_decay;
	float hierarchy_min_ratio;
	// Zero retains legacy behavior. Positive limits are independent of semantic confidence.
	float simplify_error_relative;
	float simplify_error_absolute;
	float max_axis_shrink;
	bool preserve_components;
	float base_simplify_ratio;
	float base_error_relative;
};

struct clodFeatureMetrics
{
	uint64_t input_feature_vertices = 0;
	uint64_t input_feature_tris = 0;
	uint64_t boundary_vertices = 0;
	uint64_t non_manifold_vertices = 0;
	uint64_t sharp_edge_vertices = 0;
	uint64_t boundary_components = 0;
	uint64_t sharp_ring_components = 0;
	uint64_t circular_hole_loops = 0;
	uint64_t circular_hole_vertices = 0;
	uint64_t functional_boundary_vertices = 0;
	uint64_t cylindrical_vertices = 0;
	uint64_t thin_wall_vertices = 0;
	uint64_t protected_vertices = 0;
	uint64_t critical_vertices = 0;
	uint64_t feature_importance_sum_ppm = 0;
	uint64_t feature_importance_max_ppm = 0;
};


// Borrowed mesh view passed into the LOD builder; the caller owns all pointed-to arrays.
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
	const float* feature_importance;
	const unsigned char* feature_lock;
	const unsigned int* component_ids;
	clodFeatureMetrics* feature_metrics;
#ifdef __cplusplus
	clodTimingStats* timing_stats;
#endif
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


// Output group descriptor containing hierarchy links, bounds, error metrics, and LOD level.
struct clodGroup
{
	int depth;
	clodBounds simplified;
};


typedef int (*clodOutput)(void* output_context, clodGroup group, const clodCluster* clusters, size_t cluster_count, size_t task_index, unsigned int thread_index);


typedef void (*clodIteration)(void* iteration_context, void* output_context, int depth, size_t task_count);
