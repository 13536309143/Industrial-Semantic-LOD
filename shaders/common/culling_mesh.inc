//==============================================================================
// shaders/common/culling_mesh.inc
// Mesh primitive orientation helpers shared by mesh shading paths.
// The routines keep front-face tests consistent with renderer winding and two-sided settings.
//==============================================================================
bool isFrontFacingHW(vec4 ha, vec4 hb, vec4 hc)
{
  return determinant(mat3(ha.xyw, hb.xyw, hc.xyw)) <= 0.0;
}
