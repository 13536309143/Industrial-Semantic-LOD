#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <glm/glm.hpp>

namespace lodclusters {
struct AdaptiveTwoPassSettings
{
  float motionPixels = 8.0f;
  float fastPixels = 24.0f;
  uint32_t maxInterval = 8;
};

// The deadline is conditional on outstanding motion: a stationary camera does
// not cause periodic second passes. Small steps accumulate, even if the camera
// stops before reaching the movement threshold.
class AdaptiveTwoPassController
{
public:
  enum Reason : uint32_t { Startup=1, FastMotion=2, AccumulatedMotion=4, Deadline=8, Streaming=16, Fixed=32 };
  struct Decision { bool active=false; bool bypassTemporal=false; uint32_t reason=0; float accumulatedPixels=0; uint32_t age=0; };
  void reset() { m_initialized=false; m_accumulated=0; m_age=0; }
  Decision update(float motion, bool enabled, bool adaptive, bool largeEnough, bool frozen, bool streaming,
                  const AdaptiveTwoPassSettings& settings)
  {
    Decision d;
    if(!enabled || frozen) { reset(); return d; }
    if(!adaptive) { d.active=true; d.reason=Fixed; reset(); return d; }
    motion=std::isfinite(motion) ? std::max(motion,0.0f) : 1e6f;
    if(!largeEnough && !streaming)
    {
      // A small-scene moving frame renders current-frustum candidates without
      // historical Hi-Z, so its new Hi-Z is already valid when motion stops.
      // Do not leave small stationary scenes in the uninitialized bypass state.
      d.bypassTemporal=!m_initialized || motion>0.001f;
      m_initialized=true;m_accumulated=0;m_age=0;
      return d;
    }
    m_accumulated=std::min(1e6f,m_accumulated+motion);
    ++m_age;
    if(largeEnough)
    {
      if(!m_initialized) d.reason|=Startup;
      if(motion>=std::max(settings.fastPixels,0.01f)) d.reason|=FastMotion;
      if(m_accumulated>=std::max(settings.motionPixels,0.01f)) d.reason|=AccumulatedMotion;
      if(m_accumulated>0.001f && m_age>=std::max(settings.maxInterval,1u)) d.reason|=Deadline;
    }
    // Residency can change without camera motion; retain fixed two-pass safety
    // until a residency-change signal is available to this controller.
    if(streaming) d.reason|=Streaming;
    d.active=d.reason!=0;
    d.accumulatedPixels=m_accumulated;
    d.age=m_age;
    d.bypassTemporal=!d.active && (!m_initialized || m_accumulated>0.001f || motion>0.001f);
    if(d.active) { m_initialized=true; m_accumulated=0; m_age=0; }
    return d;
  }
private:
  bool m_initialized=false;
  float m_accumulated=0;
  uint32_t m_age=0;
};

inline float projectedSceneMotion(const glm::mat4& current, const glm::mat4& previous,
                                  const glm::vec3& lo, const glm::vec3& hi, glm::vec2 viewport)
{
  float matrixDelta=0;
  for(int c=0;c<4;++c) for(int r=0;r<4;++r) matrixDelta=std::max(matrixDelta,std::abs(current[c][r]-previous[c][r]));
  if(matrixDelta<1e-7f) return 0;
  float pixels=0;
  unsigned valid=0;
  for(int x=0;x<3;++x) for(int y=0;y<3;++y) for(int z=0;z<3;++z)
  {
    const glm::vec3 p=glm::mix(lo,hi,glm::vec3(x,y,z)*0.5f);
    const glm::vec4 a=current*glm::vec4(p,1), b=previous*glm::vec4(p,1);
    if(a.w<=1e-6f || b.w<=1e-6f) continue;
    const glm::vec2 an=glm::vec2(a)/a.w, bn=glm::vec2(b)/b.w;
    // Far off-screen probes are unstable close to the eye plane. Any camera
    // motion with no usable probes takes the fast-motion safety path.
    if(glm::any(glm::greaterThan(glm::abs(an),glm::vec2(4))) || glm::any(glm::greaterThan(glm::abs(bn),glm::vec2(4)))) continue;
    const glm::vec2 delta=glm::abs(an-bn)*viewport*0.5f;
    pixels=std::max(pixels,std::max(delta.x,delta.y));
    ++valid;
  }
  return valid ? pixels : 1e6f;
}
}
