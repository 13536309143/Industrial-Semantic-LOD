#pragma once
#include "adaptive_two_pass.hpp"

namespace lodclusters {
struct AdaptiveCostSettings
{
  uint32_t probeInterval=16;
  float hysteresis=.05f;
  float smoothing=.25f;
  uint32_t warmupFrames=32;
};

// Learn the cost of two visibility-safe paths. A single moving pass bypasses
// stale Hi-Z; a dual pass recovers disocclusions against the current depth.
// Periodic probes keep the choice responsive when scene occlusion changes.
class AdaptiveCostController
{
public:
  enum Reason : uint32_t { CostDual=64, Probe=128, CostSingle=256 };
  using Decision=AdaptiveTwoPassController::Decision;
  void reset()
  {
    m_single=0;m_dual=0;m_haveSingle=false;m_haveDual=false;m_preferDual=true;m_sinceProbe=0;m_initialized=false;
    m_warmFrames=0;m_tick=0;m_lastSingle=0;m_lastDual=0;m_probeRemaining=0;m_probeDual=false;
  }
  void observe(bool dual,double milliseconds,const AdaptiveCostSettings& settings)
  {
    if(!std::isfinite(milliseconds)||milliseconds<=0||milliseconds>10000)return;
    double& value=dual?m_dual:m_single;bool& known=dual?m_haveDual:m_haveSingle;
    const double alpha=std::clamp(settings.smoothing,.01f,1.f);
    value=known ? value+(milliseconds-value)*alpha : milliseconds;known=true;
    (dual?m_lastDual:m_lastSingle)=m_tick;
  }
  Decision update(float motion,bool enabled,bool adaptive,bool largeEnough,bool frozen,bool streaming,
                  const AdaptiveCostSettings& settings)
  {
    Decision d;
    ++m_tick;
    if(!enabled||frozen) { m_initialized=false;return d; }
    if(!adaptive||streaming) { d.active=true;d.reason=streaming?16u:32u;m_initialized=true;return d; }
    motion=std::isfinite(motion)?std::max(motion,0.f):1e6f;
    if(!largeEnough)
    {
      d.bypassTemporal=!m_initialized||motion>.001f;m_initialized=true;return d;
    }
    if(m_warmFrames<std::max(settings.warmupFrames,8u)||!m_haveSingle||!m_haveDual)
    {
      // Calibrate even during the stationary application warm-up. The single
      // path deliberately bypasses Hi-Z, exactly as it does during motion.
      bool dual=m_initialized&&((m_warmFrames/4u)%2u!=0u);
      ++m_warmFrames;d.active=dual;d.bypassTemporal=!dual;
      d.reason=Probe|(dual?CostDual:CostSingle);m_initialized=true;return d;
    }
    bool desired=m_preferDual;
    if(m_haveSingle&&m_haveDual)
    {
      const double h=std::clamp(settings.hysteresis,0.f,.4f);
      if(m_dual<m_single*(1-h))desired=true;
      else if(m_single<m_dual*(1-h))desired=false;
    }
    const uint64_t alternateAge=m_tick-(desired?m_lastDual:m_lastSingle);
    if(desired!=m_preferDual&&alternateAge<=8)m_preferDual=desired;
    if(motion<=.001f)
    { d.bypassTemporal=!m_initialized;m_initialized=true;return d; }
    if(m_probeRemaining)
    {
      --m_probeRemaining;d.active=m_probeDual;d.bypassTemporal=!d.active;
      d.reason=Probe|(d.active?CostDual:CostSingle);return d;
    }
    const double confidence=std::abs(m_single-m_dual)/std::max(m_single,m_dual);
    const uint32_t interval=std::max(settings.probeInterval,1u)*(confidence>.2?4u:(confidence>.1?2u:1u));
    bool dual=m_preferDual;
    if((desired!=m_preferDual&&alternateAge>8)||++m_sinceProbe>=interval)
    {
      dual=!m_preferDual;d.reason=Probe;m_sinceProbe=0;
      m_probeDual=dual;m_probeRemaining=1;
    }
    d.active=dual;d.reason|=dual?CostDual:CostSingle;
    d.bypassTemporal=!dual;d.age=m_sinceProbe;m_initialized=true;
    return d;
  }
  double singleCost()const{return m_single;}
  double dualCost()const{return m_dual;}
private:
  double m_single=0,m_dual=0;
  bool m_haveSingle=false,m_haveDual=false,m_preferDual=true,m_initialized=false;
  uint32_t m_sinceProbe=0;
  uint32_t m_warmFrames=0,m_probeRemaining=0;
  uint64_t m_tick=0,m_lastSingle=0,m_lastDual=0;
  bool m_probeDual=false;
};
}
