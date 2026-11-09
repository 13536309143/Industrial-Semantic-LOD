//==============================================================================
// shaders/post/fullscreen.vert.glsl
// Fullscreen triangle vertex shader used by post and background passes.
// Positions are generated procedurally so no vertex buffer is required.
//==============================================================================
#version 460


layout(location = 0) out vec2 uv;


void main()
{
  uv.x        = (gl_VertexIndex == 2) ? 2.0 : 0.0;
  uv.y        = (gl_VertexIndex == 1) ? 2.0 : 0.0;
  gl_Position = vec4(uv * vec2(2.0, -2.0) + vec2(-1.0, 1.0), 0.0, 1.0);
  uv.y        = 1.0 - uv.y;
}
