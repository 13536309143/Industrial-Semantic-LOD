//==============================================================================
// src/meshlod/meshlod.cpp
// Single translation unit that includes the header-only clustered LOD implementation.
// Keeping the implementation here prevents duplicate symbols while preserving fast local helper definitions.
//==============================================================================
#include <meshoptimizer.h>
#include "meshlod.h"
#include "meshlod_bounds.h"
#include "meshlod_clustering.h"
#include "meshlod_simplify.h"
#include "meshlod_build.h"
#include "meshlod_local_indices.h"
