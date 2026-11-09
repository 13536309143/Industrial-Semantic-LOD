//==============================================================================
// shaders/common/render_shading.glsl
// Shared fragment shading, visualization, wireframe, picking, ambient-occlusion, and debug color utilities.
// Render and debug shaders use this file to keep material interpretation and readback encoding consistent.
//==============================================================================
#extension GL_EXT_fragment_shader_barycentric : enable


vec3 batlow(float t)
{

  t             = clamp(t, 0.0f, 1.0f);

  const vec3 c5 = vec3(10.741, -0.934, -16.125);

  const vec3 c4 = vec3(-28.888, 2.021, 34.529);

  const vec3 c3 = vec3(24.263, -0.335, -20.561);

  const vec3 c2 = vec3(-6.069, -1.511, 2.47);

  const vec3 c1 = vec3(0.928, 1.455, 0.327);

  const vec3 c0 = vec3(0.007, 0.103, 0.341);
  const vec3 result = ((((c5 * t + c4) * t + c3) * t + c2) * t + c1) * t + c0;

  return min(result, vec3(1.0f));
}


vec3 hue2rgb(float hue)
{

  hue= fract(hue);
  return clamp(vec3(
    abs(hue*6.0-3.0)-1.0,
    2.0-abs(hue*6.0-2.0),

    2.0-abs(hue*6.0-4.0)
  ), vec3(0), vec3(1));
}


vec3 lodMix(float v)
{
  float low = 0.15;
  if (v == 0) {
    return vec3(1);
  }
  else if (v < low) {
    return mix(vec3(1), hue2rgb(0.5), v / low);
  }
  else {
    v = (v - low) / (1.0 - low);
    return hue2rgb(0.5 - v * 0.5);
  }
}


vec3 colorizeID(uint clusterID)
{
  return vec3(unpackUnorm4x8(murmurHash(clusterID ^ view.colorXor)).xyz * 0.5 + 0.3);
}

#include "semantic_lod_palette.h"
vec3 semanticLodColor(uint instanceID)
{
  uint flags = instances[instanceID].lodPolicyFlags;
  uint priority = (flags >> SEMANTIC_LOD_PRIORITY_SHIFT) & SEMANTIC_LOD_PRIORITY_MASK;
  if((flags & SEMANTIC_LOD_VALID_BIT) == 0 || priority < 1 || priority > 10)
    return vec3(SEMANTIC_LOD_RGB_0);
  const vec3 colors[11] = vec3[11](
      vec3(SEMANTIC_LOD_RGB_0),
      vec3(SEMANTIC_LOD_RGB_1),
      vec3(SEMANTIC_LOD_RGB_2),
      vec3(SEMANTIC_LOD_RGB_3),
      vec3(SEMANTIC_LOD_RGB_4),
      vec3(SEMANTIC_LOD_RGB_5),
      vec3(SEMANTIC_LOD_RGB_6),
      vec3(SEMANTIC_LOD_RGB_7),
      vec3(SEMANTIC_LOD_RGB_8),
      vec3(SEMANTIC_LOD_RGB_9),
      vec3(SEMANTIC_LOD_RGB_10));
  return colors[priority];
}


vec3 visualizeColor(uint visData, uint instanceID)
{
  if(view.visualize == VISUALIZE_CLUSTER || view.visualize == VISUALIZE_GROUP || view.visualize == VISUALIZE_TRIANGLE)
  {
    return colorizeID(visData).xyz;
  }
  else if (view.visualize == VISUALIZE_LOD)
  {
    return vec3(lodMix(uintBitsToFloat(visData))) * 0.7 + 0.2;

  }
  else if (view.visualize == VISUALIZE_MATERIAL)
  {
    return pow(unpackUnorm4x8(instances[instanceID].packedColor).xyz * 0.95 + 0.05, vec3(1.0/2.2));
  }
  else if (view.visualize == VISUALIZE_SEMANTIC_LOD)
  {
    return semanticLodColor(instanceID);
  }
  else
  {
    return vec3(0.95);
  }
}

// Main material/debug shading path shared by clustered rendering and visualization modes.
vec4 shading(uint instanceID, vec3 wPos, vec3 wNormal, vec4 wTangent, vec2 oTexCoord, uint visData, float overheadLight, // Ray-marched ambient occlusion used for inspection rather than physically based lighting.
float ambientOcclusion
)
{
  // Policy IDs are categorical: lighting/AO must not alter their colors.
  if(view.visualize == VISUALIZE_SEMANTIC_LOD)
    return vec4(semanticLodColor(instanceID), 1.0f);

  const vec3 skyColor           = view.skyParams.skyColor;
  const vec3 groundColor        = view.skyParams.groundColor;
  const float materialRoughness = 0;

        vec3  materialAlbedo    = visualizeColor(visData, instanceID);


  vec4 color   = vec4(0.f);
  vec3 normal  = wNormal.xyz;

  vec3 wEyePos = vec3(view.viewMatrixI[3].x, view.viewMatrixI[3].y, view.viewMatrixI[3].z);

  vec3 eyeDir  = normalize(wEyePos.xyz - wPos.xyz);


#if ALLOW_VERTEX_TEXCOORDS && 0

  if (view.visualize == VISUALIZE_GREY){

  #if ALLOW_VERTEX_TANGENTS

    vec3 tangent   = normalize(wTangent.xyz);
    vec3 bitangent = cross(normal,tangent) * wTangent.w;
    vec3 procNrm;
    procNrm.xy = sin(oTexCoord.xy * 1000) * 0.25;
    procNrm.z = 1.0;

    procNrm = normalize(procNrm);

    normal = normalize(mat3(tangent, bitangent, normal) * procNrm);
  #endif

    materialAlbedo.xy *= (oTexCoord * 0.3) + 0.7;
  }
#endif


  float ambientIntensity = 1.f;
  vec3  ambientLighting  = ambientOcclusion * materialAlbedo * ambientIntensity
                         * mix(mix(groundColor, skyColor, dot(normal, view.wUpDir.xyz) * 0.5 + 0.5), vec3(0.5), 0.5) ;


  float lightMixer             = view.lightMixer;
  float flashlightIntensity    = 1.0f - lightMixer;
  float overheadLightIntensity = lightMixer;


  vec3  flashlightLighting  = vec3(0.f);
  {

    flashlightIntensity *= max(skyColor.x, max(skyColor.y, skyColor.z));

    vec3  lightDir     = normalize(view.wLightPos.xyz - wPos.xyz);
    vec3  reflDir      = normalize(-reflect(lightDir, normal));
    float bsdf         = abs(dot(normal, lightDir)) + pow(max(0, dot(reflDir, eyeDir)), 16) * 0.3;
    flashlightLighting = flashlightIntensity * materialAlbedo * bsdf;
  }


  vec3 overheadLightColor = view.skyParams.sunColor * view.skyParams.sunIntensity;

  vec3 overheadLighting   = vec3(overheadLightIntensity * overheadLight * overheadLightColor);
  {

    vec3 lightDir = normalize(view.skyParams.sunDirection);
    vec3 reflDir  = normalize(-reflect(lightDir, normal));
    float diffuse    = max(0, dot(normal, lightDir));
    float specular   = pow(max(0, dot(reflDir, eyeDir)), 16) * 0.3;
    float bsdf       = diffuse + specular;
    overheadLighting = overheadLighting * materialAlbedo * bsdf;
  }

  color.xyz = overheadLighting + flashlightLighting + ambientLighting;
  color.w   = 1.0;


  return color;
}


// Store an object identifier and depth in one 64-bit value so atomic min keeps the nearest pick.
uint64_t packPickingValue(uint32_t v, float z)
{

  z         = 1.f - clamp(z, 0.f, 1.f);

  uint bits = floatBitsToUint(z);
  bits ^= (int(bits) >> 31) | 0x80000000u;

  uint64_t value = (uint64_t(bits) << 32) | uint64_t(v);
  return value;
}


float getLineWidth(in vec3 deltas, in float thickness, in float smoothing, in vec3 barys)
{
  barys         = smoothstep(deltas * (thickness), deltas * (thickness + smoothing), barys);
  float minBary = min(barys.x, min(barys.y, barys.z));
  return 1.0 - minBary;
}


float edgePosition(vec3 barycentrics)
{
  return max(barycentrics.z, max(barycentrics.y, barycentrics.x));
}


float stipple(in float stippleRepeats, in float stippleLength, in float edgePos)
{
  float offset = 1.0 / stippleRepeats;
  offset *= 0.5 * stippleLength;
  float pattern = fract((edgePos + offset) * stippleRepeats);
  return 1.0 - step(stippleLength, pattern);
}


vec3 addWireframe(vec3 color, vec3 barycentrics, bool frontFacing, vec3 barycentricsDerivatives, vec3 wireColor)
{
  float oThickness    = view.wireThickness * 0.5;
  float thickness     = oThickness * 0.5;
  float smoothing     = oThickness * view.wireSmoothing;
  bool  enableStipple = (view.wireStipple == 1);


  float edgePos = edgePosition(barycentrics);

  if(!frontFacing)
  {
    enableStipple = true;
    wireColor     = view.wireBackfaceColor;
  }


  vec3 deltas = barycentricsDerivatives;


  float lineWidth = getLineWidth(deltas, thickness, smoothing, barycentrics);


  if(enableStipple)
  {

    float stippleFact = stipple(view.wireStippleRepeats, view.wireStippleLength, edgePos);
    lineWidth *= stippleFact;
  }


  return mix(color, wireColor, lineWidth);
}

#if SUPPORTS_RT == 1


uint wangHash(uint seed)
{
  seed = (seed ^ 61) ^ (seed >> 16);
  seed *= 9;
  seed = seed ^ (seed >> 4);
  seed *= 0x27d4eb2d;
  seed = seed ^ (seed >> 15);
  return seed;
}


uint xxhash32(uint3 p)
{

  const uvec4 primes = uint4(2246822519U, 3266489917U, 668265263U, 374761393U);
  uint        h32;
  h32 = p.z + primes.w + p.x * primes.y;
  h32 = primes.z * ((h32 << 17) | (h32 >> (32 - 17)));
  h32 += p.y * primes.y;
  h32 = primes.z * ((h32 << 17) | (h32 >> (32 - 17)));
  h32 = primes.x * (h32 ^ (h32 >> 15));
  h32 = primes.y * (h32 ^ (h32 >> 13));
  return h32 ^ (h32 >> 16);
}


uint pcg(inout uint state)
{
  uint prev = state * 747796405u + 2891336453u;
  uint word = ((prev >> ((prev >> 28u) + 4u)) ^ prev) * 277803737u;
  state     = prev;
  return (word >> 22u) ^ word;
}


float rand(inout uint seed)
{

  uint r = pcg(seed);
  return float(r) * (1.F / float(0xffffffffu));
}


void computeDefaultBasis(const vec3 z, out vec3 x, out vec3 y)
{
  const float yz = -z.y * z.z;
  y = normalize(((abs(z.z) > 0.99999f) ? vec3(-z.x * z.y, 1.0f - z.y * z.y, yz) : vec3(-z.x * z.z, yz, 1.0f - z.z * z.z)));


  x = cross(y, z);
}
#ifndef M_PI


#define M_PI 3.141592653589
#endif


vec3 offsetRay(vec3 p, vec3 dir, vec3 geonrm)
{
  vec3 n = sign(dot(dir, geonrm)) * geonrm;


  const float epsilon = 1.0f / 65536.0f;


  float magnitude = length(p);
  float offset    = epsilon * magnitude;

  vec3 offsetVector = n * offset;

  vec3 offsetPoint = p + offsetVector;
  return offsetPoint;
}


float ambientOcclusion(vec3 wPos, vec3 wNormal, uint32_t sampleCount, float radius)
{
  if (sampleCount == 0) return 0.7f;


  vec3     z    = wNormal;
  vec3     x, y;

  computeDefaultBasis(z, x, y);

  uint32_t occlusion = 0u;

  for(uint32_t i = 0; i < sampleCount; i++)
  {

    float r1 = 2 * M_PI * rand(seed);

    float r2 = rand(seed);

    float sq = sqrt(1.0 - r2);

    vec3 wDirection  = vec3(cos(r1) * sq, sin(r1) * sq, sqrt(r2));
    wDirection       = wDirection.x * x + wDirection.y * y + wDirection.z * z;
    rayHitAO         = 1.f;
    uint mask        = 0xFF;
    traceRayEXT(asScene, gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT,


                mask , 0, 0, 1, offsetRay(wPos, wDirection, wNormal), 1e-4f, wDirection, radius, 1);
    if(rayHitAO > 0.f)
    {
      occlusion++;
    }
  }

  float linearAo = float(sampleCount - occlusion) / float(sampleCount);
  return max(0.2f, linearAo* linearAo);
}


float traceShadowRay(vec3 wPos, vec3 wNormal, vec3 wDirection)
{
  rayHitAO         = 1.f;
  uint  mask       = 0xFF;
  uint  flags      = gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT;
  float minT       = 0.001f;
  float maxT       = 10000000.0f;
  traceRayEXT(asScene, flags, mask, 0, 0, 1, offsetRay(wPos, wDirection, wNormal), minT, wDirection, maxT, 1);

  return (rayHitAO > 0.f) ? 0.0F : 1.0f;
}


float determinant(vec3 a, vec3 b, vec3 c)
{
  return dot(cross(a, b), c);
}


vec3 intersectRayTriangle(vec3 origin, vec3 direction, vec3 v0, vec3 v1, vec3 v2)
{

  vec3 e1 = v1 - v0;
  vec3 e2 = v2 - v0;


  vec3 planeNormal = cross(e1, e2);


  float nDotDir = dot(planeNormal, direction);


  float t = dot(planeNormal, v0 - origin) / nDotDir;


  vec3 p = origin + t * direction;


  vec3  temp = p - v0;

  float det  = determinant(e1, e2, planeNormal);
  float u    = dot(cross(temp, e2), planeNormal) / det;
  float v    = dot(cross(e1, temp), planeNormal) / det;
  float w    = 1.0 - u - v;

  return vec3(w, u, v);
}


ivec2 objectToPixel(vec3 objectPos)
{

  vec3 wObjectPos = gl_ObjectToWorldEXT * vec4(objectPos, 1.f);


  vec4 pPos = view.viewProjMatrix * vec4(wObjectPos, 1.f);

  pPos /= pPos.w;

  pPos.xy = pPos.xy * vec2(0.5f) + vec2(0.5f);

  pPos.xy *= vec2(gl_LaunchSizeEXT.xy);
  return ivec2(pPos.xy);
}
#endif
