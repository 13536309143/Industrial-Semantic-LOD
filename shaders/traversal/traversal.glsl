//==============================================================================
// shaders/traversal/traversal.glsl
// Shared traversal math for packed task records, transform scale, and LOD screen-error tests.
// Run, presort, and setup shaders include these helpers to keep LOD decisions identical across traversal variants.
//==============================================================================
#define FLT_MAX 3.402823466e+38f


TraversalInfo unpackTraversalInfo(uint64_t packed64)
{

  u32vec2       data = unpack32(packed64);
  TraversalInfo info;
  info.instanceID = data.x;
  info.packedNode = data.y;
  return info;
}


uint64_t packTraversalInfo(TraversalInfo info)
{
  return pack64(u32vec2(info.instanceID, info.packedNode));
}


float computeUniformScale(mat4 transform)
{
  return max(max(length(vec3(transform[0])), length(vec3(transform[1]))), length(vec3(transform[2])));
}


float computeUniformScale(mat4x3 transform)
{
  return max(max(length(vec3(transform[0])), length(vec3(transform[1]))), length(vec3(transform[2])));
}


vec3 TraversalMetric_getSphere(TraversalMetric metric)
{
  return vec3(metric.boundingSphereX, metric.boundingSphereY, metric.boundingSphereZ);
}


void TraversalMetric_setSphere(inout TraversalMetric metric, vec3 sphere)
{
  metric.boundingSphereX = sphere.x;
  metric.boundingSphereY = sphere.y;
  metric.boundingSphereZ = sphere.z;
}


bool testForTraversal(mat4x3 instanceToEye, float uniformScale, TraversalMetric metric, float errorScale)
{

  vec3  boundingSpherePos = vec3(metric.boundingSphereX, metric.boundingSphereY, metric.boundingSphereZ);
  float minDistance       = view.nearPlane;
  float sphereDistance    = length(vec3(instanceToEye * vec4(boundingSpherePos, 1.0f)));

  float errorDistance     = max(minDistance, sphereDistance - metric.boundingSphereRadius * uniformScale);
  float errorOverDistance = metric.maxQuadricError * uniformScale / errorDistance;


  return errorOverDistance >= build.errorOverDistanceThreshold * errorScale;
}


float computeLodTransitionFactor(float currentError, float nextError, float threshold)
{

  float transitionStart = threshold * 0.8;
  float transitionEnd = threshold * 1.2;


  float errorRatio = (currentError + nextError) * 0.5;


  return smoothstep(transitionStart, transitionEnd, errorRatio);
}


float evaluateLodTransition(mat4x3 instanceToEye, float uniformScale, TraversalMetric currentMetric, TraversalMetric nextMetric, float errorScale)
{

  vec3  boundingSpherePos = vec3(currentMetric.boundingSphereX, currentMetric.boundingSphereY, currentMetric.boundingSphereZ);
  float minDistance       = view.nearPlane;
  float sphereDistance    = length(vec3(instanceToEye * vec4(boundingSpherePos, 1.0f)));

  float errorDistance     = max(minDistance, sphereDistance - currentMetric.boundingSphereRadius * uniformScale);

  float currentError = currentMetric.maxQuadricError * uniformScale / errorDistance;
  float nextError = nextMetric.maxQuadricError * uniformScale / errorDistance;

  return computeLodTransitionFactor(currentError, nextError, build.errorOverDistanceThreshold * errorScale);
}


bool testForTraversal(vec3 wViewPos, float uniformScale, TraversalMetric metric, float errorScale)
{

  vec3  boundingSpherePos = vec3(metric.boundingSphereX, metric.boundingSphereY, metric.boundingSphereZ);
  float minDistance       = view.nearPlane;

  float sphereDistance    = length(wViewPos - boundingSpherePos);

  float errorDistance     = max(minDistance, sphereDistance - metric.boundingSphereRadius * uniformScale);
  float errorOverDistance = metric.maxQuadricError * uniformScale / errorDistance;


  return errorOverDistance >= build.errorOverDistanceThreshold * errorScale;
}
