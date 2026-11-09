//==============================================================================
// src/meshlod/meshlod_simplify.h
// Detects geometric feature rims, budgets genuine anchors, and applies a bounded
// normal metric with a sampled geometric regression guard. All alternatives keep
// the same caller and inter-group locks; constrained builds never use sloppy fallback.
//==============================================================================
#pragma once

#include "meshlod_impl.h"
#include <numeric>
#include <map>

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

struct FeatureRim
{
    float center[3] = {}, normal[3] = {}, tangent[3] = {}, bitangent[3] = {}, radius = 0;
};

// Verify a closed, planar circular rim in its own fitted 2D plane. World-axis
// extents and distance to an unweighted centroid are not rotation/tessellation invariant.
static bool isNearCircularComponent(const std::vector<std::array<float, 3>>& positions, const std::vector<std::vector<unsigned int>>& graph, const std::vector<unsigned int>& component, FeatureRim* result = nullptr)
{
    if (component.size() < 8) return false;
    for (unsigned int v : component) if (graph[v].size() != 2) return false;
    const auto& origin = positions[*std::min_element(component.begin(), component.end())];
    float u[3] = {}, n[3] = {}; float best = 0;
    for (unsigned int v : component)
    {
        float d[3] = {positions[v][0]-origin[0], positions[v][1]-origin[1], positions[v][2]-origin[2]};
        if (lengthSq3(d) > best) { best = lengthSq3(d); memcpy(u, d, sizeof(u)); }
    }
    if (!normalize3(u)) return false;
    best = 0;
    for (unsigned int v : component)
    {
        float d[3] = {positions[v][0]-origin[0], positions[v][1]-origin[1], positions[v][2]-origin[2]};
        float c[3] = {u[1]*d[2]-u[2]*d[1], u[2]*d[0]-u[0]*d[2], u[0]*d[1]-u[1]*d[0]};
        if (lengthSq3(c) > best) { best = lengthSq3(c); memcpy(n, c, sizeof(n)); }
    }
    if (!normalize3(n)) return false;
    float vaxis[3] = {n[1]*u[2]-n[2]*u[1], n[2]*u[0]-n[0]*u[2], n[0]*u[1]-n[1]*u[0]};
    double a[3][4] = {};
    for (unsigned int v : component)
    {
        float d[3] = {positions[v][0]-origin[0], positions[v][1]-origin[1], positions[v][2]-origin[2]};
        double x=dot3(d,u), y=dot3(d,vaxis), q[3]={x,y,1.0}, r=x*x+y*y;
        for (int j=0;j<3;++j) { for (int k=0;k<3;++k) a[j][k]+=q[j]*q[k]; a[j][3]+=q[j]*r; }
    }
    for (int j=0;j<3;++j)
    {
        int pivot=j;
        for(int k=j+1;k<3;++k) if(fabs(a[k][j])>fabs(a[pivot][j])) pivot=k;
        if(fabs(a[pivot][j])<1e-20) return false;
        for(int k=j;k<4;++k) std::swap(a[pivot][k],a[j][k]);
        double scale=a[j][j];for(int k=j;k<4;++k)a[j][k]/=scale;
        for(int i=0;i<3;++i)if(i!=j){double t=a[i][j];for(int k=j;k<4;++k)a[i][k]-=t*a[j][k];}
    }
    double cx=a[0][3]*.5, cy=a[1][3]*.5, r2=a[2][3]+cx*cx+cy*cy;
    if(r2<=1e-20) return false;
    double radius=sqrt(r2);
    for (unsigned int v : component)
    {
        float d[3] = {positions[v][0]-origin[0], positions[v][1]-origin[1], positions[v][2]-origin[2]};
        double x=dot3(d,u)-cx,y=dot3(d,vaxis)-cy;
        if(fabs(dot3(d,n))>radius*.02 || fabs(sqrt(x*x+y*y)-radius)>radius*.08) return false;
    }
    if (result)
    {
        for (int axis = 0; axis < 3; ++axis)
        {
            result->center[axis] = origin[axis] + float(cx)*u[axis] + float(cy)*vaxis[axis];
            result->normal[axis] = n[axis];
            result->tangent[axis] = u[axis];
            result->bitangent[axis] = vaxis[axis];
        }
        result->radius = float(radius);
    }
    return true;
}

// A circular silhouette is not necessarily a hole. Reject filled coplanar
// disks (solid end caps and the outer rim of a perforated plate). A blind-hole
// floor at another depth is not treated as a filled mouth. This is a conservative
// geometric opening test, not a semantic guarantee of functional significance.
static bool isOpenCircularRim(const clodMesh& mesh, const FeatureRim& rim)
{
    const float probes[5][2] = {{0,0},{.65f,0},{0,.65f},{-.65f,0},{0,-.65f}};
    bool covered[5] = {}; unsigned int coverage = 0;
    for (size_t t = 0; t < mesh.index_count; t += 3)
    {
        float xy[3][2]; bool coplanar = true;
        for (int j=0;j<3;++j)
        {
            const unsigned int index=mesh.indices[t+j];
            if(index>=mesh.vertex_count){coplanar=false;break;}
            const float* p=featureVertexPosition(mesh,index);
            float d[3]={p[0]-rim.center[0],p[1]-rim.center[1],p[2]-rim.center[2]};
            if(fabsf(dot3(d,rim.normal))>rim.radius*.005f){coplanar=false;break;}
            xy[j][0]=dot3(d,rim.tangent)/rim.radius;xy[j][1]=dot3(d,rim.bitangent)/rim.radius;
        }
        if(!coplanar)continue;
        float ax=xy[1][0]-xy[0][0],ay=xy[1][1]-xy[0][1];
        float bx=xy[2][0]-xy[0][0],by=xy[2][1]-xy[0][1];
        float determinant=ax*by-ay*bx;
        if(fabsf(determinant)<1e-12f)continue;
        for(int k=0;k<5;++k)if(!covered[k])
        {
            float px=probes[k][0]-xy[0][0],py=probes[k][1]-xy[0][1];
            float u=(px*by-py*bx)/determinant,v=(ax*py-ay*px)/determinant;
            if(u>=-1e-5f&&v>=-1e-5f&&u+v<=1.00001f)
            { covered[k]=true;if(++coverage>=2)return false; }
        }
    }
    return true;
}

// Importance labels are not interpolated geometry. A scalar label attribute
// penalizes label gradients, not geometric deformation inside a labelled region.
// Use bounded, area-weighted source normals as the soft appearance metric.
static void buildFeatureAttributeSet(IterationContext& context)
{
    const clodMesh& source = context.mesh;
    const float weight = std::min(0.05f, std::max(0.0f, context.config.feature_attribute_weight) * 0.00625f)
        * std::min(2.0f, std::max(0.0f, context.config.feature_soft_scale));
    if (weight <= 0 || context.feature_normals.size() != source.vertex_count * 3 || source.attribute_count > 29)
        return;
    const size_t old_count = source.attribute_count;
    const size_t new_count = old_count + 3;
    context.feature_attributes.assign(source.vertex_count * new_count, 0.0f);
    context.feature_attribute_weights.assign(new_count, 0.0f);
    for (size_t v = 0; v < source.vertex_count; ++v)
    {
        if (old_count && source.vertex_attributes)
            memcpy(&context.feature_attributes[v * new_count], &source.vertex_attributes[v * (source.vertex_attributes_stride / sizeof(float))], old_count * sizeof(float));
        memcpy(&context.feature_attributes[v * new_count + old_count], &context.feature_normals[v * 3], 3 * sizeof(float));
    }
    if (old_count && source.attribute_weights)
        memcpy(context.feature_attribute_weights.data(), source.attribute_weights, old_count * sizeof(float));
    for (size_t i = old_count; i < new_count; ++i)
        context.feature_attribute_weights[i] = weight;
    context.mesh.vertex_attributes = context.feature_attributes.data();
    context.mesh.vertex_attributes_stride = sizeof(float) * new_count;
    context.mesh.attribute_weights = context.feature_attribute_weights.data();
    context.mesh.attribute_count = new_count;
    context.mesh.feature_attribute_count = 3;
}

static void analyzeFeatureConstraints(IterationContext& context)
{
	clodMesh& mesh = context.mesh;
    const unsigned char* supplied_feature_locks = mesh.feature_lock;
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
    context.feature_seam_protect.assign(mesh.vertex_count, 0);
    mesh.feature_user_lock = supplied_feature_locks;
	context.feature_normals.assign(mesh.vertex_count * 3, 0.0f);

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
	std::vector<unsigned int> incident_count(welded_vertex_count);
	std::vector<float> normal_sum(welded_vertex_count * 3);
	std::vector<float> min_edge_sq(welded_vertex_count, FLT_MAX);
	std::vector<float> max_edge_sq(welded_vertex_count, 0.0f);
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
		float area_normal[3] = {normal[0], normal[1], normal[2]};
		if (!normalize3(normal))
			continue;
		triangle_normals[t] = {normal[0], normal[1], normal[2]};

		unsigned int welded_tri[3] = {original_to_weld[ia], original_to_weld[ib], original_to_weld[ic]};
		if (welded_tri[0] == welded_tri[1] || welded_tri[1] == welded_tri[2] || welded_tri[2] == welded_tri[0])
			continue;

		for (int i = 0; i < 3; ++i)
		{
			unsigned int v = welded_tri[i];
			incident_count[v]++;
			normal_sum[v * 3 + 0] += area_normal[0];
			normal_sum[v * 3 + 1] += area_normal[1];
			normal_sum[v * 3 + 2] += area_normal[2];

			unsigned int n = welded_tri[(i + 1) % 3];
			const float* pv = featureTopologyPosition(welded_positions, v);
			const float* pn = featureTopologyPosition(welded_positions, n);
			float edge[3] = {pv[0] - pn[0], pv[1] - pn[1], pv[2] - pn[2]};
			min_edge_sq[v] = std::min(min_edge_sq[v], lengthSq3(edge));
			min_edge_sq[n] = std::min(min_edge_sq[n], lengthSq3(edge));
			max_edge_sq[v] = std::max(max_edge_sq[v], lengthSq3(edge));
			max_edge_sq[n] = std::max(max_edge_sq[n], lengthSq3(edge));
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
            const float cosine = dot3(n0, n1);
            if (cosine > sharp_cos && cosine < 0.990268f)
                cylindrical[edge.a] = cylindrical[edge.b] = 1;
            if (cosine < sharp_cos && cosine > -0.99999f)
			{
				sharp[edge.a] = 1;
				sharp[edge.b] = 1;
				sharp_graph[edge.a].push_back(edge.b);
				sharp_graph[edge.b].push_back(edge.a);
			}
		}
	}

    for (auto& neighbors : boundary_graph) std::sort(neighbors.begin(), neighbors.end());
    for (auto& neighbors : sharp_graph) std::sort(neighbors.begin(), neighbors.end());
	std::vector<std::vector<unsigned int>> boundary_components;
	std::vector<std::vector<unsigned int>> sharp_components;
	metrics.boundary_components = collectFeatureComponents(boundary_graph, &boundary_components);
	metrics.sharp_ring_components = collectFeatureComponents(sharp_graph, &sharp_components);

    std::vector<std::vector<unsigned int>> open_rims;
    auto classifyRims = [&](const std::vector<std::vector<unsigned int>>& components,
                            const std::vector<std::vector<unsigned int>>& graph) {
        for (const auto& component : components)
        {
            FeatureRim rim;
            if (isNearCircularComponent(welded_positions, graph, component, &rim) && isOpenCircularRim(mesh, rim))
            {
                ++metrics.circular_hole_loops;
                markComponentVertices(component, circular_hole);
                markComponentVertices(component, functional_boundary);
                open_rims.push_back(component);
            }
        }
    };
    classifyRims(boundary_components, boundary_graph);
    classifyRims(sharp_components, sharp_graph);


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

        float normal[3] = {normal_sum[w * 3], normal_sum[w * 3 + 1], normal_sum[w * 3 + 2]};
        normalize3(normal);
        for (int a = 0; a < 3; ++a)
            context.feature_normals[v * 3 + a] = normal[a];
        // Conservative small geometric-detail proxy: short AND anisotropic
        // local edges. It does not measure physical wall thickness.
        if (sharp[w] && min_edge_sq[w] < thin_edge_sq && min_edge_sq[w] < max_edge_sq[w] * 0.0025f)
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
        // Preserve the attribute seam along a detected hole mouth, while
        // permitting collapses along the rim. Protect is deliberately separate
        // from the sparse geometric Lock anchors selected below.
        if (circular_hole[w] && context.config.feature_seam_protection)
        {
            context.feature_locks[v] |= meshopt_SimplifyVertex_Protect;
            context.feature_seam_protect[v] = 1;
        }

		if (importance >= context.config.feature_protect_threshold)
			metrics.protected_vertices++;


		uint64_t ppm = uint64_t(importance * 1000000.0f + 0.5f);
		metrics.feature_importance_sum_ppm += ppm;
		metrics.feature_importance_max_ppm = std::max(metrics.feature_importance_max_ppm, ppm);
	}

    // Stable anchor coverage on closed circular feature rims. Protect marks
    // attribute seams; only Lock guarantees that an anchor cannot be collapsed.
    std::vector<std::vector<unsigned int>> rims;
    if (context.config.feature_critical_threshold <= 0.96f)
    for (const auto& component : open_rims)
    {
        std::vector<unsigned int> sorted = component;
        std::sort(sorted.begin(), sorted.end());
        rims.push_back(std::move(sorted));
    }
    const size_t budget = size_t(float(welded_vertex_count) * clampFeature01(context.config.feature_hard_lock_ratio));
    std::vector<unsigned char> anchor_mask(welded_vertex_count);
    size_t used = 0;
    std::vector<size_t> rim_counts(rims.size(), 0);
    for (size_t k = 0; k < 12 && used < budget; ++k)
        for (size_t r = 0; r < rims.size() && used < budget; ++r)
            if (k < rims[r].size()) { ++rim_counts[r]; ++used; }
    // Select equally spaced angles AFTER allocating each rim's budget.
    // Truncating a farthest-point sequence at e.g. 12 points leaves unequal gaps.
    for (size_t r = 0; r < rims.size(); ++r)
    {
        if (!rim_counts[r]) continue;
        FeatureRim rim;
        const auto& component = rims[r];
        const auto& graph = boundary[component.front()] ? boundary_graph : sharp_graph;
        if (!isNearCircularComponent(welded_positions, graph, component, &rim)) continue;
        auto angle = [&](unsigned int id) {
            const auto& p = welded_positions[id];
            float d[3] = {p[0]-rim.center[0],p[1]-rim.center[1],p[2]-rim.center[2]};
            return atan2f(dot3(d,rim.bitangent),dot3(d,rim.tangent));
        };
        const float phase = angle(component.front());
        for (size_t k = 0; k < rim_counts[r]; ++k)
        {
            const float goal = phase + 6.283185307179586f * float(k) / float(rim_counts[r]);
            unsigned int best_id = component.front(); float best_angle = FLT_MAX;
            for (unsigned int id : component)
            {
                float delta = fabsf(remainderf(angle(id)-goal,6.283185307179586f));
                if (delta < best_angle) { best_angle = delta; best_id = id; }
            }
            anchor_mask[best_id] = 1;
        }
    }
    // Spend remaining anchor budget on genuine boundary corners/junctions and
    // thin-detail extrema as well as circular openings. Never rank by vertex ID
    // alone: spatial separation prevents exhausting the budget in one corner.
    std::vector<unsigned int> candidates;
    for (unsigned int w=0; w<welded_vertex_count; ++w)
    {
        const auto& graph = boundary[w] ? boundary_graph[w] : sharp_graph[w];
        bool corner = !graph.empty() && graph.size()!=2;
        if(graph.size()==2)
        {
            float a[3],b[3];
            for(int k=0;k<3;++k){a[k]=welded_positions[graph[0]][k]-welded_positions[w][k];b[k]=welded_positions[graph[1]][k]-welded_positions[w][k];}
            if(normalize3(a)&&normalize3(b))corner=dot3(a,b)>-0.5f;
        }
        if(!anchor_mask[w]&&(corner||thin_wall[w]))candidates.push_back(w);
    }
    std::vector<double> separation(welded_vertex_count,bbox_diag_sq);
    auto update_separation=[&](unsigned int a){for(unsigned int w:candidates)
    {double d=0;for(int k=0;k<3;++k){double v=double(welded_positions[w][k])-welded_positions[a][k];d+=v*v;}separation[w]=std::min(separation[w],d);}};
    size_t actual_used=0;for(unsigned int a=0;a<anchor_mask.size();++a)if(anchor_mask[a]){++actual_used;update_separation(a);}
    while(actual_used<budget&&!candidates.empty())
    {
        size_t best=0;double best_score=-1;
        for(size_t j=0;j<candidates.size();++j)
        {unsigned int w=candidates[j];double score=separation[w]*(thin_wall[w]?1.5:1.0);if(score>best_score){best_score=score;best=j;}}
        unsigned int chosen=candidates[best];anchor_mask[chosen]=1;++actual_used;
        candidates.erase(candidates.begin()+best);update_separation(chosen);
    }
    for (size_t v = 0; v < mesh.vertex_count; ++v)
        if (anchor_mask[original_to_weld[v]])
        {
            context.feature_locks[v] |= meshopt_SimplifyVertex_Lock;
        }
    metrics.critical_vertices = 0;
    for (size_t v = 0; v < mesh.vertex_count; ++v)
    {
        if (supplied_feature_locks) context.feature_locks[v] |= supplied_feature_locks[v];
        if (context.feature_locks[v] & meshopt_SimplifyVertex_Lock) ++metrics.critical_vertices;
    }

	mesh.feature_importance = context.feature_importance.data();
	mesh.feature_lock = context.feature_locks.data();
    mesh.feature_seam_protect = context.feature_seam_protect.data();
	if (mesh.feature_metrics)
		*mesh.feature_metrics = metrics;

    // Avoid perturbing unrelated meshes when no supported critical rim exists.
    if (metrics.critical_vertices) buildFeatureAttributeSet(context);
}

// Exact point/triangle distance used by a deterministic local quality guard.
// The guard is a small sampled check, not a Hausdorff bound or a proof of quality.
static float featurePointTriangleDistanceSq(const float* p, const float* a, const float* b, const float* c)
{
    float ab[3], ac[3], ap[3], bp[3], cp[3];
    for(int k=0;k<3;++k){ab[k]=b[k]-a[k];ac[k]=c[k]-a[k];ap[k]=p[k]-a[k];bp[k]=p[k]-b[k];cp[k]=p[k]-c[k];}
    float d1=dot3(ab,ap),d2=dot3(ac,ap),d3=dot3(ab,bp),d4=dot3(ac,bp),d5=dot3(ab,cp),d6=dot3(ac,cp);
    if(d1<=0&&d2<=0)return lengthSq3(ap);
    if(d3>=0&&d4<=d3)return lengthSq3(bp);
    if(d6>=0&&d5<=d6)return lengthSq3(cp);
    float vc=d1*d4-d3*d2, vb=d5*d2-d1*d6, va=d3*d6-d5*d4;
    float u=0,v=0;
    if(vc<=0&&d1>=0&&d3<=0)u=d1/std::max(d1-d3,1e-30f);
    else if(vb<=0&&d2>=0&&d6<=0)v=d2/std::max(d2-d6,1e-30f);
    else if(va<=0&&(d4-d3)>=0&&(d5-d6)>=0){v=(d4-d3)/std::max((d4-d3)+(d5-d6),1e-30f);u=1-v;}
    else {float total=va+vb+vc;if(fabsf(total)<1e-30f)return std::min({lengthSq3(ap),lengthSq3(bp),lengthSq3(cp)});u=vb/total;v=vc/total;}
    float d[3]={ap[0]-u*ab[0]-v*ac[0],ap[1]-u*ab[1]-v*ac[1],ap[2]-u*ab[2]-v*ac[2]};
    return lengthSq3(d);
}

static double featureSampledSurfaceError(const clodMesh& mesh, const std::vector<unsigned int>& from, const std::vector<unsigned int>& to, double* normal_error = nullptr)
{
    if(from.empty()||to.empty())return from.empty()?0.0:DBL_MAX;
    const size_t triangles=from.size()/3;
    std::vector<double> area(triangles);double total=0;
    for(size_t t=0;t<triangles;++t)
    {
        const float* a=featureVertexPosition(mesh,from[t*3]);const float* b=featureVertexPosition(mesh,from[t*3+1]);const float* c=featureVertexPosition(mesh,from[t*3+2]);
        float ab[3]={b[0]-a[0],b[1]-a[1],b[2]-a[2]},ac[3]={c[0]-a[0],c[1]-a[1],c[2]-a[2]};
        float n[3]={ab[1]*ac[2]-ab[2]*ac[1],ab[2]*ac[0]-ab[0]*ac[2],ab[0]*ac[1]-ab[1]*ac[0]};
        total+=sqrt(double(lengthSq3(n)));area[t]=total;
    }
    if(total<=0)return 0;
    const size_t samples=64;double error=0, angle_error=0;size_t t=0;
    for(size_t k=0;k<samples;++k)
    {
        double goal=total*(double(k)+.5)/double(samples);
        while(t+1<triangles&&area[t]<goal)++t;
        const float* a=featureVertexPosition(mesh,from[t*3]);const float* b=featureVertexPosition(mesh,from[t*3+1]);const float* c=featureVertexPosition(mesh,from[t*3+2]);
        // Fixed barycentric sequence, shared by the compared alternatives.
        float u=float((k*37+17)%101)/101.f,v=float((k*61+29)%103)/103.f;
        if(u+v>1){u=1-u;v=1-v;}
        float p[3]={a[0]+u*(b[0]-a[0])+v*(c[0]-a[0]),a[1]+u*(b[1]-a[1])+v*(c[1]-a[1]),a[2]+u*(b[2]-a[2])+v*(c[2]-a[2])};
        float best=FLT_MAX;size_t nearest=0;
        for(size_t j=0;j<to.size();j+=3)
        {
            const float* q0=featureVertexPosition(mesh,to[j]);const float* q1=featureVertexPosition(mesh,to[j+1]);const float* q2=featureVertexPosition(mesh,to[j+2]);
            float lower=0;
            for(int axis=0;axis<3;++axis){float lo=std::min({q0[axis],q1[axis],q2[axis]}),hi=std::max({q0[axis],q1[axis],q2[axis]});float d=std::max({lo-p[axis],0.f,p[axis]-hi});lower+=d*d;}
            if(lower<best)
            {
                float distance=featurePointTriangleDistanceSq(p,q0,q1,q2);
                if(distance<best){best=distance;nearest=j;}
            }
        }
        error+=best;
        if(normal_error)
        {
            float x[3],y[3],n[3],xx[3],yy[3],nn[3];
            const float* q0=featureVertexPosition(mesh,to[nearest]);const float* q1=featureVertexPosition(mesh,to[nearest+1]);const float* q2=featureVertexPosition(mesh,to[nearest+2]);
            for(int axis=0;axis<3;++axis){x[axis]=b[axis]-a[axis];y[axis]=c[axis]-a[axis];xx[axis]=q1[axis]-q0[axis];yy[axis]=q2[axis]-q0[axis];}
            for(int axis=0;axis<3;++axis){int j=(axis+1)%3,k=(axis+2)%3;n[axis]=x[j]*y[k]-x[k]*y[j];nn[axis]=xx[j]*yy[k]-xx[k]*yy[j];}
            if(normalize3(n)&&normalize3(nn))angle_error+=std::max(0.f,1.f-dot3(n,nn));
        }
    }
    if(normal_error)*normal_error=angle_error/double(samples);
    return error/double(samples);
}

// A small per-group BVH makes deterministic triangle quadrature practical.
// Vertices and edge midpoints explicitly cover small clipped corners which a
// short area-sample sequence can miss. Validation uses independent random samples.
struct FeatureSurfaceTree
{
    struct Node { float lo[3],hi[3];size_t begin=0,end=0;int left=-1,right=-1; };
    const clodMesh& mesh;const std::vector<unsigned int>& faces;
    std::vector<size_t> order;std::vector<Node> nodes;
    FeatureSurfaceTree(const clodMesh& m,const std::vector<unsigned int>& f):mesh(m),faces(f)
    {order.resize(f.size()/3);std::iota(order.begin(),order.end(),size_t(0));nodes.reserve(order.size()*2+1);if(!order.empty())build(0,order.size());}
    int build(size_t begin,size_t end)
    {
        int id=int(nodes.size());nodes.emplace_back();Node n{};n.begin=begin;n.end=end;
        for(int a=0;a<3;++a){n.lo[a]=FLT_MAX;n.hi[a]=-FLT_MAX;}
        for(size_t k=begin;k<end;++k)for(int j=0;j<3;++j)
        {const float* p=featureVertexPosition(mesh,faces[order[k]*3+j]);for(int a=0;a<3;++a){n.lo[a]=std::min(n.lo[a],p[a]);n.hi[a]=std::max(n.hi[a],p[a]);}}
        if(end-begin>8)
        {
            int axis=0;for(int a=1;a<3;++a)if(n.hi[a]-n.lo[a]>n.hi[axis]-n.lo[axis])axis=a;
            size_t middle=(begin+end)/2;
            auto center=[&](size_t t){return featureVertexPosition(mesh,faces[t*3])[axis]+featureVertexPosition(mesh,faces[t*3+1])[axis]+featureVertexPosition(mesh,faces[t*3+2])[axis];};
            std::nth_element(order.begin()+begin,order.begin()+middle,order.begin()+end,[&](size_t a,size_t b){float x=center(a),y=center(b);return x<y||(x==y&&a<b);});
            n.left=build(begin,middle);n.right=build(middle,end);
        }
        nodes[id]=n;return id;
    }
    float lower(const float* p,int id)const
    {float e=0;for(int a=0;a<3;++a){float d=std::max({nodes[id].lo[a]-p[a],0.f,p[a]-nodes[id].hi[a]});e+=d*d;}return e;}
    void query(const float* p,int id,float& best,size_t& triangle)const
    {
        const Node& n=nodes[id];if(lower(p,id)>best)return;
        if(n.left<0)
        {
            for(size_t k=n.begin;k<n.end;++k){size_t t=order[k];float d=featurePointTriangleDistanceSq(p,featureVertexPosition(mesh,faces[t*3]),featureVertexPosition(mesh,faces[t*3+1]),featureVertexPosition(mesh,faces[t*3+2]));if(d<best){best=d;triangle=t;}}
        }
        else
        {
            int first=n.left,second=n.right;if(lower(p,first)>lower(p,second))std::swap(first,second);
            query(p,first,best,triangle);query(p,second,best,triangle);
        }
    }
};

static double featureQuadratureError(const clodMesh& mesh,const std::vector<unsigned int>& from,const FeatureSurfaceTree& to,double* normal_error,double importance_scale=4.)
{
    *normal_error=0;if(from.empty()||to.faces.empty())return from.empty()?0:DBL_MAX;
    static const float bary[7][3]={{1,0,0},{0,1,0},{0,0,1},{.5f,.5f,0},{0,.5f,.5f},{.5f,0,.5f},{1.f/3,1.f/3,1.f/3}};
    // Interior area quadrature for the global non-regression gate. Boundary
    // fidelity has its own arc-length measure and must not bias this integral.
    static const float interior[7][3]={{1.f/3,1.f/3,1.f/3},{.059715872f,.470142064f,.470142064f},{.470142064f,.059715872f,.470142064f},{.470142064f,.470142064f,.059715872f},{.797426985f,.101286507f,.101286507f},{.101286507f,.797426985f,.101286507f},{.101286507f,.101286507f,.797426985f}};
    static const double interior_weight[7]={.225,.132394152788506,.132394152788506,.132394152788506,.125939180544827,.125939180544827,.125939180544827};
    double total=0,error=0,angle_error=0;
    for(size_t i=0;i<from.size();i+=3)
    {
        const float* a=featureVertexPosition(mesh,from[i]);const float* b=featureVertexPosition(mesh,from[i+1]);const float* c=featureVertexPosition(mesh,from[i+2]);
        float x[3],y[3],n[3];for(int k=0;k<3;++k){x[k]=b[k]-a[k];y[k]=c[k]-a[k];}
        for(int k=0;k<3;++k)n[k]=x[(k+1)%3]*y[(k+2)%3]-x[(k+2)%3]*y[(k+1)%3];
        double area=sqrt(double(lengthSq3(n)));if(area<=0)continue;normalize3(n);
        double importance=mesh.feature_importance?std::max({mesh.feature_importance[from[i]],mesh.feature_importance[from[i+1]],mesh.feature_importance[from[i+2]]}):0.;total+=area*(1.+importance_scale*importance);
        for(int sample=0;sample<7;++sample)
        {
            const float* coordinates=importance_scale==0?interior[sample]:bary[sample];
            float p[3];for(int k=0;k<3;++k)p[k]=coordinates[0]*a[k]+coordinates[1]*b[k]+coordinates[2]*c[k];
            float d=FLT_MAX;size_t t=0;to.query(p,0,d,t);double importance=mesh.feature_importance?std::max({mesh.feature_importance[from[i]],mesh.feature_importance[from[i+1]],mesh.feature_importance[from[i+2]]}):0.;double w=area*(importance_scale==0?interior_weight[sample]:(sample==6?.5:1./12.))*(1.+importance_scale*importance);error+=w*d;
            const float* q0=featureVertexPosition(mesh,to.faces[t*3]);const float* q1=featureVertexPosition(mesh,to.faces[t*3+1]);const float* q2=featureVertexPosition(mesh,to.faces[t*3+2]);
            float xx[3],yy[3],nn[3];for(int k=0;k<3;++k){xx[k]=q1[k]-q0[k];yy[k]=q2[k]-q0[k];}
            for(int k=0;k<3;++k)nn[k]=xx[(k+1)%3]*yy[(k+2)%3]-xx[(k+2)%3]*yy[(k+1)%3];
            if(normalize3(nn))angle_error+=w*std::max(0.f,1.f-dot3(n,nn));
        }
    }
    if(!total)return 0;*normal_error=angle_error/total;return error/total;
}

// Measure the complete boundary/crease curves, not only a handful of anchor
// positions. Weld position seams locally so authored normal seams are not
// mistaken for open geometric edges. Segment BVH uses degenerate triangles.
struct FeatureTopology { long long euler=0;bool manifold=true; };
static std::vector<unsigned int> featureCurveSegments(const clodMesh& mesh,const std::vector<unsigned int>& faces,FeatureTopology* topology=nullptr)
{
    std::map<std::array<float,3>,unsigned int> welded;
    std::unordered_map<unsigned int,unsigned int> remap;
    std::unordered_map<uint64_t,FeatureEdgeInfo> edges;
    std::vector<std::array<float,3>> normals(faces.size()/3);
    for(size_t t=0;t<faces.size();t+=3)
    {
        unsigned int tri[3];
        for(int j=0;j<3;++j)
        {
            unsigned int v=faces[t+j];auto found=remap.find(v);
            if(found==remap.end())
            {
                const float* p=featureVertexPosition(mesh,v);auto value=welded.emplace(std::array<float,3>{p[0],p[1],p[2]},v);
                tri[j]=value.first->second;remap.emplace(v,tri[j]);
            }
            else tri[j]=found->second;
        }
        const float* a=featureVertexPosition(mesh,tri[0]);const float* b=featureVertexPosition(mesh,tri[1]);const float* c=featureVertexPosition(mesh,tri[2]);
        float x[3],y[3],n[3];for(int k=0;k<3;++k){x[k]=b[k]-a[k];y[k]=c[k]-a[k];}
        for(int k=0;k<3;++k)n[k]=x[(k+1)%3]*y[(k+2)%3]-x[(k+2)%3]*y[(k+1)%3];
        if(!normalize3(n))continue;
        normals[t/3]={n[0],n[1],n[2]};
        for(int j=0;j<3;++j)
        {
            auto& e=edges[featureEdgeKey(tri[j],tri[(j+1)%3])];
            if(e.count==0){e.a=std::min(tri[j],tri[(j+1)%3]);e.b=std::max(tri[j],tri[(j+1)%3]);e.first_tri=unsigned(t/3);}
            else if(e.count==1)e.second_tri=unsigned(t/3);
            ++e.count;
        }
    }
    std::vector<unsigned int> segments;std::vector<uint64_t> keys;
    for(const auto& entry:edges)
    {
        const auto& e=entry.second;
        if(e.count==1||(e.count==2&&dot3(normals[e.first_tri].data(),normals[e.second_tri].data())<0.57357644f))keys.push_back(entry.first);
    }
    std::sort(keys.begin(),keys.end());
    for(uint64_t key:keys){const auto& e=edges[key];segments.insert(segments.end(),{e.a,e.b,e.b});}
    if(topology)
    {
        topology->euler=static_cast<long long>(welded.size())-static_cast<long long>(edges.size())+static_cast<long long>(faces.size()/3);
        topology->manifold=true;for(const auto& e:edges)if(e.second.count>2)topology->manifold=false;
    }
    return segments;
}

// Directional support is a cheap envelope/thin-dimension safeguard, not a
// measurement of physical wall thickness. Axes plus diagonals avoid relying
// exclusively on the original axis-aligned bounding box.
static double featureSupportError(const clodMesh& mesh,const std::vector<unsigned int>& source,const std::vector<unsigned int>& candidate)
{
    double maximum=0.;
    for(int x=-1;x<=1;++x)for(int y=-1;y<=1;++y)for(int z=-1;z<=1;++z)
    {
        if(x<0||(x==0&&y<0)||(x==0&&y==0&&z<=0))continue;
        double source_lo=DBL_MAX,source_hi=-DBL_MAX,candidate_lo=DBL_MAX,candidate_hi=-DBL_MAX;
        for(unsigned int v:source){const float* p=featureVertexPosition(mesh,v);double q=x*double(p[0])+y*double(p[1])+z*double(p[2]);source_lo=std::min(source_lo,q);source_hi=std::max(source_hi,q);}
        for(unsigned int v:candidate){const float* p=featureVertexPosition(mesh,v);double q=x*double(p[0])+y*double(p[1])+z*double(p[2]);candidate_lo=std::min(candidate_lo,q);candidate_hi=std::max(candidate_hi,q);}
        double change=std::max(fabs(source_lo-candidate_lo),fabs(source_hi-candidate_hi));
        maximum=std::max(maximum,change*change/double(x*x+y*y+z*z));
    }
    return maximum;
}

static double featureCurveError(const clodMesh& mesh,const std::vector<unsigned int>& source,const FeatureSurfaceTree& candidate,double* tail=nullptr)
{
    if(tail)*tail=source.empty()?0.:DBL_MAX;
    if(source.empty())return 0.;
    if(candidate.faces.empty())return DBL_MAX;
    double total=0,error=0;std::vector<std::pair<double,double>> distribution;
    for(size_t i=0;i<source.size();i+=3)
    {
        const float* a=featureVertexPosition(mesh,source[i]);const float* b=featureVertexPosition(mesh,source[i+1]);
        double length=0;for(int k=0;k<3;++k)length+=double(b[k]-a[k])*(b[k]-a[k]);length=sqrt(length);
        for(int s=0;s<3;++s)
        {
            float p[3];for(int k=0;k<3;++k)p[k]=a[k]+float(s)*.5f*(b[k]-a[k]);
            float distance=FLT_MAX;size_t nearest=0;candidate.query(p,0,distance,nearest);
            double weight=length*(s==1?.5:.25);total+=weight;error+=weight*distance;
            if(tail)distribution.emplace_back(distance,weight);
        }
    }
    if(tail&&total>0)
    {
        std::sort(distribution.begin(),distribution.end());double cumulative=0;
        for(const auto& sample:distribution){cumulative+=sample.second;if(cumulative>=total*.95){*tail=sample.first;break;}}
    }
    return total>0?error/total:0.;
}

// A fixed percentage of all vertices can leave a small opening with only
// two or three anchors. Add a separate, budget-aware whole-rim proposal.
// It is accepted only by the same surface/curve/topology gates as every other
// candidate, never by spending additional triangles or relaxing caller locks.
static std::vector<unsigned char> featureWholeRimLocks(const clodMesh& mesh,const std::vector<unsigned int>& indices,
    const std::vector<unsigned int>& curves,const std::vector<unsigned char>& caller,size_t triangle_budget,float budget_ratio)
{
    if(!mesh.feature_seam_protect)return {};
    std::unordered_map<unsigned int,std::vector<unsigned int>> graph;
    for(size_t i=0;i<curves.size();i+=3)
    {
        unsigned int a=curves[i],b=curves[i+1];
        if(mesh.feature_seam_protect[a]&&mesh.feature_seam_protect[b]){graph[a].push_back(b);graph[b].push_back(a);}
    }
    std::vector<unsigned int> starts;for(const auto& e:graph)starts.push_back(e.first);std::sort(starts.begin(),starts.end());
    std::unordered_map<unsigned int,bool> visited;std::vector<std::vector<unsigned int>> rims;
    for(unsigned int start:starts)
    {
        if(visited[start])continue;
        std::vector<unsigned int> pending{start},component;visited[start]=true;bool closed=true;
        while(!pending.empty())
        {
            unsigned int v=pending.back();pending.pop_back();component.push_back(v);closed&=graph[v].size()==2;
            for(unsigned int n:graph[v])if(!visited[n]){visited[n]=true;pending.push_back(n);}
        }
        if(!closed||component.size()<8)continue;
        unsigned int first=*std::min_element(component.begin(),component.end()),previous=~0u,current=first;
        std::vector<unsigned int> ordered;
        do
        {
            ordered.push_back(current);const auto& neighbors=graph[current];
            unsigned int next=previous==~0u?std::min(neighbors[0],neighbors[1]):neighbors[0]==previous?neighbors[1]:neighbors[0];
            previous=current;current=next;
        }while(current!=first&&ordered.size()<=component.size());
        if(current==first&&ordered.size()==component.size())rims.push_back(std::move(ordered));
    }
    const size_t budget=std::min(size_t(64),size_t(double(triangle_budget)*std::clamp(double(budget_ratio),0.,.5)));
    if(rims.empty()||budget<rims.size()*8)return {};
    std::vector<size_t> counts(rims.size(),8);size_t used=rims.size()*8;
    for(size_t k=8;k<24&&used<budget;++k)
        for(size_t r=0;r<rims.size()&&used<budget;++r)if(k<rims[r].size()){++counts[r];++used;}
    std::map<std::array<float,3>,bool> selected;
    for(size_t r=0;r<rims.size();++r)
    {
        const auto& ring=rims[r];std::vector<double> cumulative(ring.size()+1,0.);
        for(size_t i=0;i<ring.size();++i)
        {
            const float* a=featureVertexPosition(mesh,ring[i]);const float* b=featureVertexPosition(mesh,ring[(i+1)%ring.size()]);
            double d=0;for(int k=0;k<3;++k)d+=double(a[k]-b[k])*(a[k]-b[k]);cumulative[i+1]=cumulative[i]+sqrt(d);
        }
        for(size_t k=0;k<counts[r];++k)
        {
            double target=cumulative.back()*double(k)/double(counts[r]);size_t i=size_t(std::lower_bound(cumulative.begin(),cumulative.end(),target)-cumulative.begin());
            if(i>0&&target-cumulative[i-1]<cumulative[i]-target)--i;
            const float* p=featureVertexPosition(mesh,ring[i%ring.size()]);selected[{p[0],p[1],p[2]}]=true;
        }
    }
    auto locks=caller;
    for(unsigned int v:indices)
    {
        const float* p=featureVertexPosition(mesh,v);
        if(selected.find({p[0],p[1],p[2]})!=selected.end())locks[v]|=meshopt_SimplifyVertex_Lock;
    }
    return locks;
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
        options |= meshopt_SimplifyPermissive;
		lod.resize(meshopt_simplifyWithAttributes(&lod[0], &indices[0], indices.size(),
		    mesh.vertex_positions, mesh.vertex_count, mesh.vertex_positions_stride,
		    mesh.vertex_attributes, mesh.vertex_attributes_stride, mesh.attribute_weights, mesh.attribute_count,
    active_locks, target_count, config.simplify_error_absolute > 0 ? config.simplify_error_absolute : FLT_MAX, options | meshopt_SimplifyPermissive, error));
	}

    if (config.feature_constraints && !config.feature_quality_guard && mesh.feature_attribute_count > 0 && mesh.feature_attribute_count <= mesh.attribute_count)
    {
        std::vector<unsigned int> geometric(indices.size());float geometric_error=0;
        geometric.resize(meshopt_simplifyWithAttributes(geometric.data(),indices.data(),indices.size(),
            mesh.vertex_positions,mesh.vertex_count,mesh.vertex_positions_stride,
            mesh.vertex_attributes,mesh.vertex_attributes_stride,mesh.attribute_weights,
            mesh.attribute_count-mesh.feature_attribute_count,active_locks,target_count,
            config.simplify_error_absolute>0?config.simplify_error_absolute:FLT_MAX,options,&geometric_error));
        if (geometric.size() <= lod.size() && geometric != lod)
        {
            const double scale=meshopt_simplifyScale(mesh.vertex_positions,mesh.vertex_count,mesh.vertex_positions_stride);
            const double noise=scale*scale*1e-14;
            double baseline_error=featureSampledSurfaceError(mesh,indices,geometric)+featureSampledSurfaceError(mesh,geometric,indices);
            double protected_error=featureSampledSurfaceError(mesh,indices,lod)+featureSampledSurfaceError(mesh,lod,indices);
            if (protected_error > baseline_error*1.02+noise)
            {
                lod.swap(geometric);*error=geometric_error;
            }
        }
    }

    // Industrial CAD inputs often split every faceted normal at a rim. In
    // that case Protect can make the entire rim topologically complex and halt
    // reduction. It is a soft seam hint, so relax ONLY our hint when needed to
    // approach the requested budget. Caller flags, group borders and genuine
    // Lock anchors remain unchanged; original caller normal attributes remain.
    if (lod.size() > target_count && config.feature_constraints && mesh.feature_seam_protect)
    {
        std::vector<unsigned char> budget_locks(active_locks, active_locks + mesh.vertex_count);
        bool changed = false;
        for (unsigned int v : indices)
            if (mesh.feature_seam_protect[v])
            {
                const unsigned char before = budget_locks[v];
                budget_locks[v] &= ~meshopt_SimplifyVertex_Protect;
                budget_locks[v] |= locks[v] & meshopt_SimplifyVertex_Protect;
                if (mesh.feature_user_lock) budget_locks[v] |= mesh.feature_user_lock[v];
                changed |= before != budget_locks[v];
            }
        if (changed)
        {
            std::vector<unsigned int> budget(indices.size()); float budget_error = 0;
            budget.resize(meshopt_simplifyWithAttributes(budget.data(), indices.data(), indices.size(),
                mesh.vertex_positions, mesh.vertex_count, mesh.vertex_positions_stride,
                mesh.vertex_attributes, mesh.vertex_attributes_stride, mesh.attribute_weights,
                mesh.attribute_count - mesh.feature_attribute_count, budget_locks.data(), target_count,
                config.simplify_error_absolute > 0 ? config.simplify_error_absolute : FLT_MAX, options, &budget_error));
            if (budget.size() < lod.size()) { lod.swap(budget); *error = budget_error; }
        }
    }


    // Automatic features are budget-sensitive proposals. Compare with a true
    // feature-off baseline; explicit caller engineering locks remain mandatory.
    if(config.feature_constraints && config.feature_quality_guard && mesh.feature_lock)
    {
        std::vector<unsigned char> caller_locks=locks;
        if(mesh.feature_user_lock)
            for(unsigned int v:indices)caller_locks[v]|=mesh.feature_user_lock[v];
        bool automatic=mesh.feature_attribute_count>0;
        for(unsigned int v:indices)automatic|=(mesh.feature_lock[v]&~caller_locks[v])!=0||(mesh.feature_importance&&mesh.feature_importance[v]>0);
        if(automatic)
        {
            const size_t original_attributes=mesh.attribute_count-mesh.feature_attribute_count;
            clodMesh plain=mesh;
            plain.attribute_count=original_attributes;plain.feature_attribute_count=0;
            plain.feature_lock=mesh.feature_user_lock;plain.feature_seam_protect=nullptr;
            plain.feature_importance=nullptr;
            clodConfig baseline_config=config;baseline_config.feature_constraints=false;
            baseline_config.simplify_fallback_sloppy=false;
            float ordinary_error=0;
            auto ordinary=simplify(baseline_config,plain,indices,locks,target_count,&ordinary_error);
            auto generate=[&](const std::vector<unsigned char>& flags,size_t attributes,unsigned int candidate_options,float& result_error){
                std::vector<unsigned int> result(indices.size());
                result.resize(meshopt_simplifyWithAttributes(result.data(),indices.data(),indices.size(),
                    mesh.vertex_positions,mesh.vertex_count,mesh.vertex_positions_stride,
                    mesh.vertex_attributes,mesh.vertex_attributes_stride,mesh.attribute_weights,attributes,
                    flags.data(),ordinary.size(),config.simplify_error_absolute>0?config.simplify_error_absolute:FLT_MAX,candidate_options,&result_error));
                return result;
            };
            if(!ordinary.empty())
            {
                FeatureSurfaceTree reference_tree(mesh,indices),ordinary_tree(mesh,ordinary);
                double baseline_normal0=0,baseline_normal1=0;
                const double baseline_error=featureQuadratureError(mesh,indices,ordinary_tree,&baseline_normal0,0.)
                    +featureQuadratureError(mesh,ordinary,reference_tree,&baseline_normal1,0.);
                const double baseline_normal=baseline_normal0+baseline_normal1;
                const double normal_limit=baseline_normal*1.02+1e-7;
                const double scale=meshopt_simplifyScale(mesh.vertex_positions,mesh.vertex_count,mesh.vertex_positions_stride);
                const double noise=scale*scale*1e-14;
                FeatureTopology reference_topology,ordinary_topology;
                const auto reference_curves=featureCurveSegments(mesh,indices,&reference_topology);
                const auto ordinary_curves=featureCurveSegments(mesh,ordinary,&ordinary_topology);
                FeatureSurfaceTree ordinary_curve_tree(mesh,ordinary_curves);
                double baseline_tail=0;
                const double baseline_curve=featureCurveError(mesh,reference_curves,ordinary_curve_tree,&baseline_tail);
                const double baseline_support=featureSupportError(mesh,indices,ordinary);
                double best_score=3.;std::vector<unsigned int> best=ordinary;float best_bound=ordinary_error;
                auto consider=[&](const std::vector<unsigned int>& candidate,float bound){
                    if(candidate.empty()||candidate.size()>ordinary.size()||candidate==best)return;
                    if(featureSupportError(mesh,indices,candidate)>baseline_support+noise)return;
                    if(config.preserve_components&&mesh.component_ids)
                    {
                        std::vector<unsigned int> before,after;
                        for(size_t i=0;i<indices.size();i+=3)before.push_back(mesh.component_ids[indices[i]]);
                        for(size_t i=0;i<candidate.size();i+=3)after.push_back(mesh.component_ids[candidate[i]]);
                        auto unique=[](std::vector<unsigned int>& ids){std::sort(ids.begin(),ids.end());ids.erase(std::unique(ids.begin(),ids.end()),ids.end());};
                        unique(before);unique(after);if(before!=after)return;
                    }
                    FeatureSurfaceTree tree(mesh,candidate);double normal0=0,normal1=0;
                    double distance=featureQuadratureError(mesh,indices,tree,&normal0,0.)+featureQuadratureError(mesh,candidate,reference_tree,&normal1,0.);
                    double normal=normal0+normal1;
                    if(distance>baseline_error*1.00001+noise||normal>normal_limit)return;
                    FeatureTopology topology;const auto curves=featureCurveSegments(mesh,candidate,&topology);
                    if(reference_topology.manifold&&ordinary_topology.manifold&&topology.manifold&&
                       std::llabs(topology.euler-reference_topology.euler)>std::llabs(ordinary_topology.euler-reference_topology.euler))return;
                    // A manifold source cannot earn an improvement by creating
                    // non-manifold edges when the baseline did not create them.
                    if(reference_topology.manifold&&ordinary_topology.manifold&&!topology.manifold)return;
                    FeatureSurfaceTree curve_tree(mesh,curves);
                    double tail=0;double curve=featureCurveError(mesh,reference_curves,curve_tree,&tail);
                    if(curve>baseline_curve*1.00001+noise||tail>baseline_tail*1.00001+noise)return;
                    // Normalize to the feature-off result. Neither geometry nor
                    // curve fidelity may be traded away for the other objective.
                    double score=(distance+noise)/(baseline_error+noise)+(curve+noise)/(baseline_curve+noise)+(tail+noise)/(baseline_tail+noise);
                    if(score<best_score-1e-5){best=candidate;best_score=score;best_bound=bound;}
                };
                consider(lod,*error);
                // Sparse anchors without extra normal/seam objectives.
                auto hard_locks=caller_locks;
                for(unsigned int v:indices)hard_locks[v]|=mesh.feature_lock[v]&meshopt_SimplifyVertex_Lock;
                float hard_error=0;auto hard=generate(hard_locks,original_attributes,options,hard_error);consider(hard,hard_error);
                // Soft guidance and alternative QEM regularization provide
                // genuinely different candidates; the same independent gates
                // decide whether any proposal earns its extra construction cost.
                if(mesh.feature_attribute_count)
                {float soft_error=0;auto soft=generate(caller_locks,mesh.attribute_count,options,soft_error);consider(soft,soft_error);}
                unsigned int alternate_options=options^meshopt_SimplifyRegularize;
                float alternate_error=0;auto alternate=generate(caller_locks,original_attributes,alternate_options,alternate_error);consider(alternate,alternate_error);
                float protected_error=0;auto protected_alternate=generate(hard_locks,original_attributes,alternate_options,protected_error);consider(protected_alternate,protected_error);
                const auto rim_locks=config.feature_hard_lock_ratio>0?featureWholeRimLocks(mesh,indices,reference_curves,caller_locks,ordinary.size()/3,config.feature_rim_budget_ratio):std::vector<unsigned char>{};
                if(!rim_locks.empty())
                {
                    float rim_error=0;auto ring=generate(rim_locks,original_attributes,options,rim_error);consider(ring,rim_error);
                    float alternate_rim_error=0;auto alternate_ring=generate(rim_locks,original_attributes,alternate_options,alternate_rim_error);consider(alternate_ring,alternate_rim_error);
                }
                lod.swap(best);*error=best_bound;
            }
        }
    }

	if (lod.size() > target_count && config.simplify_fallback_sloppy && !config.feature_constraints && config.simplify_error_absolute <= 0)
	{
		simplifyFallback(lod, mesh, indices, feature_locks.empty() ? locks : feature_locks, target_count, error);
		*error *= config.simplify_error_factor_sloppy;
	}

    // A locked position can become unreferenced when its incident fan is
    // removed as degenerate. Check the final surface, not just collapse flags.
    const unsigned char* mandatory_features=(config.feature_constraints&&config.feature_quality_guard)?mesh.feature_user_lock:mesh.feature_lock;
    if((mandatory_features||!locks.empty()) && !lod.empty())
    {
        std::vector<std::array<float,3>> represented;
        represented.reserve(lod.size());
        for(unsigned int v:lod){const float* p=featureVertexPosition(mesh,v);represented.push_back({p[0],p[1],p[2]});}
        std::sort(represented.begin(),represented.end());
        for(unsigned int v:indices)if(((mandatory_features?mandatory_features[v]:0)|locks[v])&meshopt_SimplifyVertex_Lock)
        {
            const float* p=featureVertexPosition(mesh,v);std::array<float,3> point={p[0],p[1],p[2]};
            if(!std::binary_search(represented.begin(),represented.end(),point))
            { *error=0.f;return indices; }
        }
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
