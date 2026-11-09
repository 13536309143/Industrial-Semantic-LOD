//==============================================================================
// src/meshlod/meshlod_config.h
// Default configuration helpers for clustered LOD generation.
// The defaults choose practical meshoptimizer limits while leaving scene code free to override feature and error weights.
//==============================================================================
#pragma once


#include "meshlod_types.h"

#ifdef __cplusplus
extern "C"
{
#endif


clodConfig clodDefaultConfig(size_t max_triangles);

#ifdef __cplusplus
}
#endif
