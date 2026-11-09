//==============================================================================
// shaders/common/culling_constants.inc
// Shared culling mode constants.
// These flags select which conservative tests are compiled into traversal and render shaders.
//==============================================================================
const float c_epsilon    = 1.2e-07f;


const float c_depthNudge = 2.0/float(1<<24);
