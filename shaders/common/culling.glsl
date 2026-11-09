//==============================================================================
// shaders/common/culling.glsl
// Shared culling include that assembles frustum, Hi-Z, and mesh-facing tests.
// Traversal and render shaders include this file to keep conservative visibility decisions consistent.
//==============================================================================
#include "culling_constants.inc"
#include "culling_frustum.inc"
#include "culling_hiz.inc"
