//==============================================================================
// src/meshlod/meshlod_bounds.h
// Computes and merges spatial/error bounds for clusters and LOD groups.
// Bounds drive screen-space error tests, frustum tests, and hierarchy construction.
//==============================================================================
#pragma once


#include "meshlod_impl.h"

namespace clod
{


clodBounds boundsCompute(const clodMesh& mesh, const std::vector<unsigned int>& indices, float error)
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


clodBounds boundsMerge(const std::vector<Cluster>& clusters, const std::vector<int>& group)
{


	std::vector<clodBounds> bounds(group.size());
	for (size_t j = 0; j < group.size(); ++j)
		bounds[j] = clusters[group[j]].bounds;
	meshopt_Bounds merged = meshopt_computeSphereBounds(&bounds[0].center[0], bounds.size(), sizeof(clodBounds), &bounds[0].radius, sizeof(clodBounds));
	clodBounds result = {};
	result.center[0] = merged.center[0];
	result.center[1] = merged.center[1];
	result.center[2] = merged.center[2];
	result.radius = merged.radius;
	result.error = 0.f;
	for (size_t j = 0; j < group.size(); ++j)

		result.error = std::max(result.error, clusters[group[j]].bounds.error);

	return result;
}

}
