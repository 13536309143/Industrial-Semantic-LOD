//==============================================================================
// shaders/common/culling_frustum.inc
// Frustum and screen-extent culling helpers.
// The functions return clip-space bounds that later stages reuse for Hi-Z queries and LOD heuristics.
//==============================================================================
bool intersectSize(vec4 clipMin, vec4 clipMax, float threshold)
{


  vec2 rect = (clipMax.xy - clipMin.xy) * 0.5 * view.viewportf.xy;

  vec2 clipThreshold = vec2(threshold);

  return any(greaterThan(rect,clipThreshold));
}


vec4 getClip(vec4 hPos, out bool valid) {

  // A box crossing the eye/near plane cannot use a corner-projected
  // screen rectangle as a conservative Hi-Z bound. Keep it for clipping.
  valid = hPos.w > c_epsilon && hPos.z <= hPos.w;

  return vec4(hPos.xyz / abs(hPos.w), hPos.w);
}


uint getCullBits(vec4 hPos)
{
  uint cullBits = 0;
  cullBits |= hPos.x < -hPos.w ?  1 : 0;
  cullBits |= hPos.x >  hPos.w ?  2 : 0;
  cullBits |= hPos.y < -hPos.w ?  4 : 0;
  cullBits |= hPos.y >  hPos.w ?  8 : 0;
  cullBits |= hPos.z <  0      ? 16 : 0;
  cullBits |= hPos.z >  hPos.w ? 32 : 0;
  cullBits |= hPos.w <= 0      ? 64 : 0;
  return cullBits;
}


vec4 getBoxCorner(vec3 bboxMin, vec3 bboxMax, int n)
{

  bvec3 useMax = bvec3((n & 1) != 0, (n & 2) != 0, (n & 4) != 0);
  return vec4(mix(bboxMin, bboxMax, useMax),1);
}


bool intersectFrustum(mat4 viewProjMatrix, vec3 bboxMin, vec3 bboxMax, mat4x3 worldTM, out vec4 oClipmin, out vec4 oClipmax, out bool oClipvalid)
{


  mat4 worldViewProjTM = viewProjMatrix * toMat4(worldTM);
  bool valid;


  vec4 hPos     = worldViewProjTM * getBoxCorner(bboxMin, bboxMax, 0);

  vec4 clip     = getClip(hPos, valid);

  uint bits     = getCullBits(hPos);

  vec4 clipMin  = clip;
  vec4 clipMax  = clip;
  bool clipValid = valid;

  [[unroll]]
  for (int n = 1; n < 8; n++){

    hPos  = worldViewProjTM * getBoxCorner(bboxMin, bboxMax, n);

    clip  = getClip(hPos, valid);


    bits &= getCullBits(hPos);


    clipMin = min(clipMin,clip);

    clipMax = max(clipMax,clip);

    clipValid = clipValid && valid;
  }

  oClipvalid = clipValid;

  oClipmin = vec4(clamp(clipMin.xy, vec2(-1), vec2(1)), clipMin.zw);
  oClipmax = vec4(clamp(clipMax.xy, vec2(-1), vec2(1)), clipMax.zw);


  return bits == 0;
}
