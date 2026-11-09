//==============================================================================
// src/meshlod/meshlod_clustering.h
// Builds meshlet-sized clusters, partitions them into groups, and locks group boundaries before simplification.
// Boundary locks preserve topology across levels so generated LOD groups can be traversed independently at runtime.
//==============================================================================
#pragma once


#include "meshlod_impl.h"

namespace clod
{


std::vector<Cluster> clusterize(const clodConfig& config, const clodMesh& mesh, const unsigned int* indices, size_t index_count)
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


	std::vector<Cluster> clusters(meshlets.size());

	for (size_t i = 0; i < meshlets.size(); ++i)
	{
		const meshopt_Meshlet& meshlet = meshlets[i];
		if (config.optimize_clusters)

			meshopt_optimizeMeshlet(&meshlet_vertices[meshlet.vertex_offset], &meshlet_triangles[meshlet.triangle_offset], meshlet.triangle_count, meshlet.vertex_count);
		clusters[i].vertices = meshlet.vertex_count;

		clusters[i].indices.resize(meshlet.triangle_count * 3);
		for (size_t j = 0; j < meshlet.triangle_count * 3; ++j)
			clusters[i].indices[j] = meshlet_vertices[meshlet.vertex_offset + meshlet_triangles[meshlet.triangle_offset + j]];
		clusters[i].group = -1;
		clusters[i].refined = -1;
	}

	return clusters;
}


std::vector<std::vector<int> > partition(const clodConfig& config, const clodMesh& mesh, const std::vector<Cluster>& clusters, const std::vector<int>& pending, const std::vector<unsigned int>& remap)
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


void lockBoundary(std::vector<unsigned char>& locks, const std::vector<std::vector<int> >& groups, const std::vector<Cluster>& clusters, const std::vector<unsigned int>& remap, const unsigned char* vertex_lock)
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

}
