//==============================================================================
// src/meshlod/meshlod.h
// C++ convenience facade over the clustered LOD builder.
// Callers provide mesh data and receive groups through callbacks without depending on the internal implementation headers.
//==============================================================================
#pragma once


#include "meshlod_types.h"
#include "meshlod_config.h"

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


		static int output(void* output_context, clodGroup group, const clodCluster* clusters, size_t cluster_count, size_t task_index, unsigned int thread_index)
		{
			return (*static_cast<Output*>(output_context))(group, clusters, cluster_count);
		}
	};

	return clodBuild(config, mesh, &output, &Call::output, nullptr);
}
#endif
