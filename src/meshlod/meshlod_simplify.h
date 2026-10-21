//==============================================================================
// src/meshlod/meshlod_simplify.h
// Runs feature-aware mesh simplification on clustered LOD input.
// The analysis computes semantic and geometric locks, then falls back to baseline meshoptimizer simplification when constraints are too tight.
//==============================================================================
#pragma once

#include "meshlod_impl.h"

namespace clod
{

static float clampFeature01(float v)
{
	return std::max(0.0f, std::min(1.0f, v));
}

static float dot3(const float* a, const float* b)
{
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

static float lengthSq3(const float* v)
{
	return dot3(v, v);
}

static bool normalize3(float* v)
{
	float len = sqrtf(lengthSq3(v));
	if (len <= 1e-20f)
		return false;

	v[0] /= len;
	v[1] /= len;
	v[2] /= len;
	return true;
}

struct FeatureEdgeInfo
{
	unsigned int a = 0;
	unsigned int b = 0;
	unsigned int first_tri = ~0u;
	unsigned int second_tri = ~0u;
	unsigned int count = 0;
};

struct SemanticLockCandidate
{
	float score = 0.0f;
	unsigned int vertex = 0;
};

static uint64_t featureEdgeKey(unsigned int a, unsigned int b)
{
	unsigned int lo = std::min(a, b);
	unsigned int hi = std::max(a, b);
	return (uint64_t(lo) << 32) | uint64_t(hi);
}

static const float* featureVertexPosition(const clodMesh& mesh, unsigned int v)
{
	return &mesh.vertex_positions[v * (mesh.vertex_positions_stride / sizeof(float))];
}

static const float* featureTopologyPosition(const std::vector<std::array<float, 3>>& positions, unsigned int v)
{
	return positions[v].data();
}

static uint64_t markComponentVertices(const std::vector<unsigned int>& component, std::vector<unsigned char>& flags)
{
	uint64_t marked = 0;
	for (unsigned int v : component)
	{
		if (!flags[v])
		{
			flags[v] = 1;
			marked++;
		}
	}
	return marked;
}

static size_t collectFeatureComponents(const std::vector<std::vector<unsigned int>>& graph, std::vector<std::vector<unsigned int>>* out_components = nullptr)
{
	size_t component_count = 0;
	std::vector<unsigned char> visited(graph.size());
	std::queue<unsigned int> queue;

	for (size_t start = 0; start < graph.size(); ++start)
	{
		if (visited[start] || graph[start].empty())
			continue;

		component_count++;
		if (out_components)
			out_components->push_back({});

		visited[start] = 1;
		queue.push(unsigned(start));

		while (!queue.empty())
		{
			unsigned int v = queue.front();
			queue.pop();

			if (out_components)
				out_components->back().push_back(v);

			for (unsigned int next : graph[v])
			{
				if (!visited[next])
				{
					visited[next] = 1;
					queue.push(next);
				}
			}
		}
	}

	return component_count;
}

static bool isNearCircularComponent(const std::vector<std::array<float, 3>>& positions, const std::vector<std::vector<unsigned int>>& graph, const std::vector<unsigned int>& component)
{
	if (component.size() < 8)
		return false;

	size_t degree2 = 0;
	float centroid[3] = {};
	float lo[3] = {FLT_MAX, FLT_MAX, FLT_MAX};
	float hi[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};

	for (unsigned int v : component)
	{
		const float* p = featureTopologyPosition(positions, v);
		centroid[0] += p[0];
		centroid[1] += p[1];
		centroid[2] += p[2];
		for (int c = 0; c < 3; ++c)
		{
			lo[c] = std::min(lo[c], p[c]);
			hi[c] = std::max(hi[c], p[c]);
		}
		if (graph[v].size() == 2)
			degree2++;
	}

	if (degree2 * 4 < component.size() * 3)
		return false;

	centroid[0] /= float(component.size());
	centroid[1] /= float(component.size());
	centroid[2] /= float(component.size());

	float extents[3] = {hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]};
	std::sort(extents, extents + 3);
	if (extents[1] <= 1e-8f || extents[2] / extents[1] > 1.35f)
		return false;

	float mean_radius = 0.0f;
	std::vector<float> radii;
	radii.reserve(component.size());

	for (unsigned int v : component)
	{
		const float* p = featureTopologyPosition(positions, v);
		float d[3] = {p[0] - centroid[0], p[1] - centroid[1], p[2] - centroid[2]};
		float r = sqrtf(lengthSq3(d));
		radii.push_back(r);
		mean_radius += r;
	}

	mean_radius /= float(radii.size());
	if (mean_radius <= 1e-8f)
		return false;

	float max_deviation = 0.0f;
	for (float r : radii)
		max_deviation = std::max(max_deviation, fabsf(r - mean_radius));

	return max_deviation / mean_radius < 0.22f;
}

static void buildFeatureAttributeSet(IterationContext& context)
{
	const clodMesh& source = context.mesh;
	if (!source.feature_importance || context.feature_importance.empty())
		return;

	size_t old_count = source.attribute_count;
	size_t new_count = old_count + 1;
	context.feature_attributes.assign(source.vertex_count * new_count, 0.0f);
	context.feature_attribute_weights.assign(new_count, 0.0f);

	if (old_count && source.vertex_attributes)
	{
		size_t old_stride = source.vertex_attributes_stride / sizeof(float);
		for (size_t v = 0; v < source.vertex_count; ++v)
			memcpy(&context.feature_attributes[v * new_count], &source.vertex_attributes[v * old_stride], sizeof(float) * old_count);
	}

	if (old_count && source.attribute_weights)
		memcpy(context.feature_attribute_weights.data(), source.attribute_weights, sizeof(float) * old_count);

	for (size_t v = 0; v < source.vertex_count; ++v)
		context.feature_attributes[v * new_count + old_count] = source.feature_importance[v];

	context.feature_attribute_weights[old_count] = std::max(0.0f, context.config.feature_attribute_weight);
	context.mesh.vertex_attributes = context.feature_attributes.data();
	context.mesh.vertex_attributes_stride = sizeof(float) * new_count;
	context.mesh.attribute_weights = context.feature_attribute_weights.data();
	context.mesh.attribute_count = new_count;
}

static void analyzeFeatureConstraints(IterationContext& context)
{
	clodMesh& mesh = context.mesh;
	clodFeatureMetrics metrics = {};

	if (!context.config.feature_constraints || mesh.vertex_count == 0 || mesh.index_count == 0)
	{
		if (mesh.feature_metrics)
			*mesh.feature_metrics = metrics;
		return;
	}

	const size_t triangle_count = mesh.index_count / 3;
	metrics.input_feature_vertices = mesh.vertex_count;
	metrics.input_feature_tris = triangle_count;

	context.feature_importance.assign(mesh.vertex_count, 0.0f);
	context.feature_locks.assign(mesh.vertex_count, 0);

	std::vector<unsigned int> representative_to_weld(mesh.vertex_count, ~0u);
	std::vector<unsigned int> original_to_weld(mesh.vertex_count, ~0u);
	std::vector<unsigned int> welded_to_representative;
	welded_to_representative.reserve(mesh.vertex_count);

	for (size_t v = 0; v < mesh.vertex_count; ++v)
	{
		unsigned int representative = context.remap.empty() ? unsigned(v) : context.remap[v];
		if (representative >= mesh.vertex_count)
			representative = unsigned(v);

		unsigned int& welded = representative_to_weld[representative];
		if (welded == ~0u)
		{
			welded = unsigned(welded_to_representative.size());
			welded_to_representative.push_back(representative);
		}
		original_to_weld[v] = welded;
	}

	const size_t welded_vertex_count = welded_to_representative.size();
	std::vector<std::array<float, 3>> welded_positions(welded_vertex_count);
	for (size_t w = 0; w < welded_vertex_count; ++w)
	{
		const float* p = featureVertexPosition(mesh, welded_to_representative[w]);
		welded_positions[w] = {p[0], p[1], p[2]};
	}

	std::vector<unsigned char> boundary(welded_vertex_count);
	std::vector<unsigned char> non_manifold(welded_vertex_count);
	std::vector<unsigned char> sharp(welded_vertex_count);
	std::vector<unsigned char> circular_hole(welded_vertex_count);
	std::vector<unsigned char> functional_boundary(welded_vertex_count);
	std::vector<unsigned char> cylindrical(welded_vertex_count);
	std::vector<unsigned char> thin_wall(welded_vertex_count);
	std::vector<SemanticLockCandidate> lock_candidates;
	std::vector<unsigned int> incident_count(welded_vertex_count);
	std::vector<float> normal_sum(welded_vertex_count * 3);
	std::vector<float> min_edge_sq(welded_vertex_count, FLT_MAX);
	std::vector<std::vector<unsigned int>> boundary_graph(welded_vertex_count);
	std::vector<std::vector<unsigned int>> sharp_graph(welded_vertex_count);
	std::vector<std::array<float, 3>> triangle_normals(triangle_count);
	std::unordered_map<uint64_t, FeatureEdgeInfo> edges;

	edges.reserve(triangle_count * 3);

	float bbox_lo[3] = {FLT_MAX, FLT_MAX, FLT_MAX};
	float bbox_hi[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
	for (size_t v = 0; v < welded_vertex_count; ++v)
	{
		const float* p = featureTopologyPosition(welded_positions, unsigned(v));
		for (int c = 0; c < 3; ++c)
		{
			bbox_lo[c] = std::min(bbox_lo[c], p[c]);
			bbox_hi[c] = std::max(bbox_hi[c], p[c]);
		}
	}

	float bbox_diag_vec[3] = {bbox_hi[0] - bbox_lo[0], bbox_hi[1] - bbox_lo[1], bbox_hi[2] - bbox_lo[2]};
	const float bbox_diag_sq = std::max(lengthSq3(bbox_diag_vec), 1e-12f);
	const float thin_edge_sq = bbox_diag_sq * 0.000004f;

	for (size_t t = 0; t < triangle_count; ++t)
	{
		unsigned int ia = mesh.indices[t * 3 + 0];
		unsigned int ib = mesh.indices[t * 3 + 1];
		unsigned int ic = mesh.indices[t * 3 + 2];
		if (ia >= mesh.vertex_count || ib >= mesh.vertex_count || ic >= mesh.vertex_count)
			continue;

		const float* a = featureVertexPosition(mesh, ia);
		const float* b = featureVertexPosition(mesh, ib);
		const float* c = featureVertexPosition(mesh, ic);
		float ab[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
		float ac[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
		float normal[3] = {
		    ab[1] * ac[2] - ab[2] * ac[1],
		    ab[2] * ac[0] - ab[0] * ac[2],
		    ab[0] * ac[1] - ab[1] * ac[0],
		};
		normalize3(normal);
		triangle_normals[t] = {normal[0], normal[1], normal[2]};

		unsigned int welded_tri[3] = {original_to_weld[ia], original_to_weld[ib], original_to_weld[ic]};
		if (welded_tri[0] == welded_tri[1] || welded_tri[1] == welded_tri[2] || welded_tri[2] == welded_tri[0])
			continue;

		for (int i = 0; i < 3; ++i)
		{
			unsigned int v = welded_tri[i];
			incident_count[v]++;
			normal_sum[v * 3 + 0] += normal[0];
			normal_sum[v * 3 + 1] += normal[1];
			normal_sum[v * 3 + 2] += normal[2];

			unsigned int n = welded_tri[(i + 1) % 3];
			const float* pv = featureTopologyPosition(welded_positions, v);
			const float* pn = featureTopologyPosition(welded_positions, n);
			float edge[3] = {pv[0] - pn[0], pv[1] - pn[1], pv[2] - pn[2]};
			min_edge_sq[v] = std::min(min_edge_sq[v], lengthSq3(edge));
		}

		for (int e = 0; e < 3; ++e)
		{
			unsigned int va = welded_tri[e];
			unsigned int vb = welded_tri[(e + 1) % 3];
			uint64_t key = featureEdgeKey(va, vb);
			FeatureEdgeInfo& edge = edges[key];
			if (edge.count == 0)
			{
				edge.a = std::min(va, vb);
				edge.b = std::max(va, vb);
				edge.first_tri = unsigned(t);
			}
			else if (edge.count == 1)
			{
				edge.second_tri = unsigned(t);
			}
			edge.count++;
		}
	}

	const float sharp_cos = cosf(55.0f * 3.14159265358979323846f / 180.0f);
	for (const auto& entry : edges)
	{
		const FeatureEdgeInfo& edge = entry.second;
		if (edge.count == 1)
		{
			boundary[edge.a] = 1;
			boundary[edge.b] = 1;
			boundary_graph[edge.a].push_back(edge.b);
			boundary_graph[edge.b].push_back(edge.a);
		}
		else if (edge.count > 2)
		{
			non_manifold[edge.a] = 1;
			non_manifold[edge.b] = 1;
		}
		else if (edge.first_tri != ~0u && edge.second_tri != ~0u)
		{
			const float* n0 = triangle_normals[edge.first_tri].data();
			const float* n1 = triangle_normals[edge.second_tri].data();
			if (dot3(n0, n1) < sharp_cos)
			{
				sharp[edge.a] = 1;
				sharp[edge.b] = 1;
				sharp_graph[edge.a].push_back(edge.b);
				sharp_graph[edge.b].push_back(edge.a);
			}
		}
	}

	std::vector<std::vector<unsigned int>> boundary_components;
	std::vector<std::vector<unsigned int>> sharp_components;
	metrics.boundary_components = collectFeatureComponents(boundary_graph, &boundary_components);
	metrics.sharp_ring_components = collectFeatureComponents(sharp_graph, &sharp_components);

	for (const std::vector<unsigned int>& component : boundary_components)
	{
		if (isNearCircularComponent(welded_positions, boundary_graph, component))
		{
			metrics.circular_hole_loops++;
			markComponentVertices(component, circular_hole);
			markComponentVertices(component, functional_boundary);
		}
		else if (component.size() <= 96)
		{
			markComponentVertices(component, functional_boundary);
		}
	}

	for (const std::vector<unsigned int>& component : sharp_components)
	{
		size_t degree2 = 0;
		for (unsigned int v : component)
			degree2 += sharp_graph[v].size() == 2 ? 1 : 0;

		if (component.size() >= 6 && degree2 * 4 >= component.size() * 3)
			markComponentVertices(component, functional_boundary);
	}

	lock_candidates.reserve(mesh.vertex_count / 8 + 1);

	for (size_t v = 0; v < mesh.vertex_count; ++v)
	{
		unsigned int w = original_to_weld[v];

		if (boundary[w])
			metrics.boundary_vertices++;
		if (non_manifold[w])
			metrics.non_manifold_vertices++;
		if (sharp[w])
			metrics.sharp_edge_vertices++;
		if (circular_hole[w])
			metrics.circular_hole_vertices++;
		if (functional_boundary[w])
			metrics.functional_boundary_vertices++;

		float normal_len = sqrtf(normal_sum[w * 3 + 0] * normal_sum[w * 3 + 0] + normal_sum[w * 3 + 1] * normal_sum[w * 3 + 1]
		                         + normal_sum[w * 3 + 2] * normal_sum[w * 3 + 2]);
		float normal_coherence = incident_count[w] ? normal_len / float(incident_count[w]) : 1.0f;

		if (!boundary[w] && !sharp[w] && incident_count[w] >= 4 && normal_coherence < 0.94f && normal_coherence > 0.35f)
			cylindrical[w] = 1;

		if ((boundary[w] || sharp[w] || functional_boundary[w]) && min_edge_sq[w] < thin_edge_sq)
			thin_wall[w] = 1;

		if (cylindrical[w])
			metrics.cylindrical_vertices++;
		if (thin_wall[w])
			metrics.thin_wall_vertices++;

		float importance = 0.0f;
		if (sharp[w])
			importance = std::max(importance, 0.55f);
		if (cylindrical[w])
			importance = std::max(importance, 0.62f);
		if (boundary[w])
			importance = std::max(importance, 0.70f);
		if (thin_wall[w])
			importance = std::max(importance, 0.78f);
		if (functional_boundary[w])
			importance = std::max(importance, 0.84f);
		if (circular_hole[w])
			importance = std::max(importance, 0.96f);
		if (non_manifold[w])
			importance = 1.0f;

		importance = clampFeature01(importance * std::max(0.0f, context.config.feature_soft_scale));
		importance = clampFeature01(importance);
		context.feature_importance[v] = importance;

		if (importance >= context.config.feature_protect_threshold)
			metrics.protected_vertices++;

		if (non_manifold[w])
		{
			context.feature_locks[v] = meshopt_SimplifyVertex_Protect;
			metrics.critical_vertices++;
		}
		else if (circular_hole[w] || importance >= context.config.feature_critical_threshold
		         || (context.config.semantic_priority > 0 && (functional_boundary[w] || thin_wall[w])))
		{
			float score = importance;
			if (circular_hole[w])
				score += 0.35f;
			if (functional_boundary[w])
				score += 0.20f;
			if (thin_wall[w])
				score += 0.15f;
			if (cylindrical[w])
				score += 0.10f;

			lock_candidates.push_back({score, unsigned(v)});
		}

		uint64_t ppm = uint64_t(importance * 1000000.0f + 0.5f);
		metrics.feature_importance_sum_ppm += ppm;
		metrics.feature_importance_max_ppm = std::max(metrics.feature_importance_max_ppm, ppm);
	}

	size_t hard_lock_budget = size_t(float(mesh.vertex_count) * std::max(0.0f, context.config.feature_hard_lock_ratio) + 0.5f);
	hard_lock_budget = std::max(hard_lock_budget, size_t(metrics.critical_vertices));
	std::sort(lock_candidates.begin(), lock_candidates.end(), [](const SemanticLockCandidate& a, const SemanticLockCandidate& b) {
		return a.score > b.score;
	});

	for (const SemanticLockCandidate& candidate : lock_candidates)
	{
		if (metrics.critical_vertices >= hard_lock_budget)
			break;

		if (candidate.vertex >= context.feature_locks.size() || context.feature_locks[candidate.vertex] & meshopt_SimplifyVertex_Protect)
			continue;

		context.feature_locks[candidate.vertex] = meshopt_SimplifyVertex_Protect;
		metrics.critical_vertices++;
	}

	mesh.feature_importance = context.feature_importance.data();
	mesh.feature_lock = context.feature_locks.data();
	if (mesh.feature_metrics)
		*mesh.feature_metrics = metrics;

	buildFeatureAttributeSet(context);
}

void simplifyFallback(std::vector<unsigned int>& lod, const clodMesh& mesh, const std::vector<unsigned int>& indices, const std::vector<unsigned char>& locks, size_t target_count, float* error)
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

std::vector<unsigned int> simplify(const clodConfig& config, const clodMesh& mesh, const std::vector<unsigned int>& indices, const std::vector<unsigned char>& locks, size_t target_count, float* error)
{
	if (target_count > indices.size())
		return indices;

	std::vector<unsigned int> lod(indices.size());
	std::vector<unsigned char> feature_locks;
	const unsigned char* active_locks = &locks[0];

	if (mesh.feature_lock)
	{
		feature_locks = locks;
		for (size_t i = 0; i < indices.size(); ++i)
		{
			unsigned int v = indices[i];
			assert(v < mesh.vertex_count);
			feature_locks[v] |= mesh.feature_lock[v];
		}
		active_locks = feature_locks.data();
	}

	unsigned int options = meshopt_SimplifySparse | meshopt_SimplifyErrorAbsolute | (config.simplify_permissive ? meshopt_SimplifyPermissive : 0) | (config.simplify_regularize ? meshopt_SimplifyRegularize : 0);

	lod.resize(meshopt_simplifyWithAttributes(&lod[0], &indices[0], indices.size(),
	    mesh.vertex_positions, mesh.vertex_count, mesh.vertex_positions_stride,
	    mesh.vertex_attributes, mesh.vertex_attributes_stride, mesh.attribute_weights, mesh.attribute_count,
    active_locks, target_count, config.simplify_error_absolute > 0 ? config.simplify_error_absolute : FLT_MAX, options, error));

	if (lod.size() > target_count && config.simplify_fallback_permissive && !config.simplify_permissive)
	{
		lod.resize(meshopt_simplifyWithAttributes(&lod[0], &indices[0], indices.size(),
		    mesh.vertex_positions, mesh.vertex_count, mesh.vertex_positions_stride,
		    mesh.vertex_attributes, mesh.vertex_attributes_stride, mesh.attribute_weights, mesh.attribute_count,
    active_locks, target_count, config.simplify_error_absolute > 0 ? config.simplify_error_absolute : FLT_MAX, options | meshopt_SimplifyPermissive, error));
	}

	if (lod.size() > target_count && config.simplify_fallback_sloppy && config.simplify_error_absolute <= 0)
	{
		simplifyFallback(lod, mesh, indices, feature_locks.empty() ? locks : feature_locks, target_count, error);
		*error *= config.simplify_error_factor_sloppy;
	}

	if (config.max_axis_shrink > 0 && !indices.empty())
	{
		float lo[3] = {FLT_MAX, FLT_MAX, FLT_MAX}, hi[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
		float outLo[3] = {FLT_MAX, FLT_MAX, FLT_MAX}, outHi[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
		for(unsigned int v : indices)
			for(int a = 0; a < 3; ++a)
			{
				float p = featureVertexPosition(mesh, v)[a];
				lo[a] = std::min(lo[a], p); hi[a] = std::max(hi[a], p);
			}
		for(unsigned int v : lod)
			for(int a = 0; a < 3; ++a)
			{
				float p = featureVertexPosition(mesh, v)[a];
				outLo[a] = std::min(outLo[a], p); outHi[a] = std::max(outHi[a], p);
			}
		float maxExtent = std::max({hi[0]-lo[0], hi[1]-lo[1], hi[2]-lo[2]});
		for(int a = 0; a < 3; ++a)
			if(hi[a]-lo[a] > maxExtent * 1e-6f && (lod.empty() || outHi[a]-outLo[a] < (hi[a]-lo[a]) * (1.0f-config.max_axis_shrink)))
			{
				*error = 0.0f;
				return indices;
			}
	}
	if (config.preserve_components && mesh.component_ids)
	{
		std::vector<unsigned int> before, after;
		before.reserve(indices.size()/3); after.reserve(lod.size()/3);
		for(size_t i = 0; i < indices.size(); i += 3) before.push_back(mesh.component_ids[indices[i]]);
		for(size_t i = 0; i < lod.size(); i += 3) after.push_back(mesh.component_ids[lod[i]]);
		std::sort(before.begin(), before.end()); before.erase(std::unique(before.begin(), before.end()), before.end());
		std::sort(after.begin(), after.end()); after.erase(std::unique(after.begin(), after.end()), after.end());
		if(before != after)
		{
			*error = 0.0f;
			return indices;
		}
	}
	if (config.simplify_error_edge_limit > 0 && config.simplify_error_absolute <= 0)
	{
		float max_edge_sq = 0;
		size_t positions_stride = mesh.vertex_positions_stride / sizeof(float);

		for (size_t i = 0; i < indices.size(); i += 3)
		{
			unsigned int a = indices[i + 0], b = indices[i + 1], c = indices[i + 2];
			assert(a < mesh.vertex_count && b < mesh.vertex_count && c < mesh.vertex_count);

			const float* va = &mesh.vertex_positions[a * positions_stride];
			const float* vb = &mesh.vertex_positions[b * positions_stride];
			const float* vc = &mesh.vertex_positions[c * positions_stride];

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

}
