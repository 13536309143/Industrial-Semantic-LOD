#include <volk.h>
#include <nvvk/barriers.hpp>
#include <zstd.h>
#include <fmt/format.h>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <stdexcept>
#include <thread>
#include <chrono>
#include "lodclusters.hpp"

namespace lodclusters {

void LodClusters::benchmarkPrepare()
{
  if(m_benchSlots.empty())
  {
    std::filesystem::create_directories(m_benchOutput);
    std::ifstream input(m_benchTrajectory);
    std::string line;
    std::getline(input, line);
    while(std::getline(input, line))
    {
      std::replace(line.begin(), line.end(), ',', ' ');
      std::istringstream row(line);
      uint32_t index;
      BenchCamera camera;
      if(row >> index >> camera.segment >> camera.eye.x >> camera.eye.y >> camera.eye.z
             >> camera.center.x >> camera.center.y >> camera.center.z
             >> camera.up.x >> camera.up.y >> camera.up.z >> camera.fov >> camera.nearPlane >> camera.farPlane)
      {
        if(index != m_benchCameras.size()) throw std::runtime_error("Nonconsecutive benchmark camera indices");
        m_benchCameras.push_back(camera);
      }
    }
    if(m_benchCameras.size() != m_benchFrames) throw std::runtime_error("Benchmark trajectory length mismatch");
    if(!m_benchDecisionsFile.empty())
    {
      std::ifstream decisions(m_benchDecisionsFile);
      std::getline(decisions,line);
      std::vector<std::string> headers;
      {std::istringstream row(line);std::string value;while(std::getline(row,value,','))headers.push_back(value);}
      auto column=[&](const std::string& name){auto it=std::find(headers.begin(),headers.end(),name);if(it==headers.end())throw std::runtime_error("Missing decision column");return size_t(it-headers.begin());};
      size_t fi=column("frame"),di=column("two_pass"),bi=column("temporal_bypass"),ri=column("trigger_reason");
      while(std::getline(decisions,line))
      {
        std::vector<std::string> values;std::istringstream row(line);std::string value;
        while(std::getline(row,value,','))values.push_back(value);
        if(values.size()<=std::max({fi,di,bi,ri}))throw std::runtime_error("Invalid decision row");
        if(std::stoul(values[fi])!=m_benchDecisions.size())throw std::runtime_error("Nonconsecutive decision frames");
        m_benchDecisions.push_back({uint32_t(std::stoul(values[di])),uint32_t(std::stoul(values[bi])),uint32_t(std::stoul(values[ri]))});
      }
      if(m_benchDecisions.size()!=m_benchFrames)throw std::runtime_error("Decision trace length mismatch");
    }
    if(m_tweak.supersample != 1) throw std::runtime_error("Benchmark requires supersample 1");
    if(m_benchCapture && m_resources.m_frameBuffer.depthStencilFormat != VK_FORMAT_D32_SFLOAT_S8_UINT)
      throw std::runtime_error("Benchmark depth capture requires D32_SFLOAT_S8_UINT");
    const uint32_t slots=m_resources.getFrameCycleCount();
    m_benchSlots.resize(slots);
    VkQueryPoolCreateInfo info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    info.queryType=VK_QUERY_TYPE_TIMESTAMP;
    info.queryCount=slots*24;
    NVVK_CHECK(vkCreateQueryPool(m_app->getDevice(), &info, nullptr, &m_resources.benchmarkQueries));
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(m_app->getPhysicalDevice(), &properties);
    m_benchPeriod=properties.limits.timestampPeriod/1e6;
    const VkDeviceSize bytes=VkDeviceSize(m_windowSize.width)*m_windowSize.height*8;
    if(m_benchCapture)
      for(auto& slot:m_benchSlots)
        NVVK_CHECK(m_resources.m_allocator.createBuffer(slot.image, bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_AUTO_PREFER_HOST, VMA_ALLOCATION_CREATE_MAPPED_BIT|VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT));
    m_benchCsv.open(std::filesystem::path(m_benchOutput)/"frames.csv");
    m_benchCsv << "frame,segment,gpu_ms,traversal_ms,second_stage_ms,cpu_interval_ms,two_pass,temporal_bypass,trigger_reason,rendered_triangles,rendered_clusters,traversal_tasks,pass0_clusters,pass1_clusters,pass0_tasks,pass1_tasks,assembly_rejected_instances,low_detail_instances,queued_root_instances,motion_pixels,captured,assembly_ms,initialization_ms,hierarchy_ms,initialized_instances,compacted_candidates,initialization_invocations,stream_geometry_upload_bytes,stream_all_upload_bytes,stream_loads,stream_unloads,stream_unserved_batch,stream_reject_transfer,stream_reject_capacity,stream_reject_slots,stream_used_bytes,stream_reserved_bytes,stream_capacity_bytes,stream_persistent_bytes,stream_resident_groups,geometry_used_bytes,geometry_reserved_bytes,operations_bytes,p1_omitted_instances\n";
    m_benchCsv << std::fixed << std::setprecision(9);
    std::ofstream meta(std::filesystem::path(m_benchOutput)/"scene.json");
    const auto& lo=m_scene->m_bbox.lo;const auto& hi=m_scene->m_bbox.hi;
    meta << fmt::format("{{\"width\":{},\"height\":{},\"instances\":{},\"assembly_nodes\":{},\"total_clusters\":{},\"bbox_lo\":[{},{},{}],\"bbox_hi\":[{},{},{}],\"warmup\":{},\"frames\":{},\"capture\":{},\"assemblyprefilter\":{},\"lowdetailfastpath\":{},\"boundedassemblylod\":{},\"hizocclusion\":{},\"twopassculling\":{},\"adaptivetwopassculling\":{},\"adaptivetwopassmode\":{},\"loderror\":{},\"semanticruntimestrength\":{},\"streaming\":{},\"depth_format\":\"D32_SFLOAT_S8_UINT\",\"frame_encoding\":\"zstd byteplanes: 4 float32 reversed-depth byte planes followed by 4 RGBA8 byte planes\"}}\n",
      m_windowSize.width,m_windowSize.height,m_scene->m_instances.size(),m_scene->m_assemblyNodes.size(),m_scene->m_totalClustersCount,
      lo.x,lo.y,lo.z,hi.x,hi.y,hi.z,m_benchWarmup,m_benchFrames,m_benchCapture,
      m_rendererConfig.useAssemblyPrefilter,m_rendererConfig.useLowDetailFastPath,m_rendererConfig.boundAssemblyLod,
      m_rendererConfig.useHizOcclusion,m_rendererConfig.useTwoPassCulling,m_rendererConfig.useAdaptiveTwoPassCulling,
      m_rendererConfig.adaptiveTwoPassMode,m_frameConfig.lodPixelError,m_rendererConfig.semanticRuntimeStrength,m_tweak.useStreaming);
    std::ofstream p1Meta(std::filesystem::path(m_benchOutput)/"p1_config.json");
    p1Meta << fmt::format("{{\"enabled\":{},\"culling_enabled\":{},\"threshold_pixels\":{}}}\n",
                         m_rendererConfig.useP1GeometryOmission, m_rendererConfig.useCulling, m_rendererConfig.p1OmissionPixels);
    std::ofstream compactMeta(std::filesystem::path(m_benchOutput)/"compact_config.json");
    compactMeta << fmt::format("{{\"assemblycompact\":{},\"assemblyearlyexit\":{},\"timestamp_count\":24}}\n",m_rendererConfig.useAssemblyCompact,m_rendererConfig.useAssemblyEarlyExit);
    LOGI("BENCHMARK_READY,%s,%u,%u\n",m_benchOutput.c_str(),m_benchWarmup,m_benchFrames);
  }
  const uint32_t index=m_resources.getFrameCycleIndex();
  benchmarkConsume(index);
  const int64_t frame=m_benchTick-int64_t(m_benchWarmup);
  if(frame==0 && m_benchColdStreaming && m_renderScene->useStreaming) {
    m_renderScene->streamingReset();
    m_benchLastTime=0;
    LOGI("BENCHMARK_COLD_GPU_RESET\n");
  }
  m_resources.benchmarkForcedTwoPass=-1;
  if(!m_benchDecisions.empty() && frame>=0 && frame<int64_t(m_benchFrames))
  {
    const auto& decision=m_benchDecisions[size_t(frame)];
    m_resources.benchmarkForcedTwoPass=int32_t(decision.dual);
    m_resources.benchmarkForcedBypass=decision.bypass;
    m_resources.benchmarkForcedReason=decision.reason;
  }
  const auto& camera=m_benchCameras[size_t(std::clamp<int64_t>(frame,0,int64_t(m_benchFrames)-1))];
  m_info.cameraManipulator->setCamera({camera.eye,camera.center,camera.up,camera.fov});
  m_info.cameraManipulator->setClipPlanes({camera.nearPlane,camera.farPlane});
  auto& slot=m_benchSlots[index];
  slot.frame=frame;slot.segment=camera.segment;
  double now=m_clock.getSeconds();
  if(m_benchPaceMs>0 && m_benchLastTime>0) {
    const double wait=m_benchLastTime+double(m_benchPaceMs)/1000.0-now;
    if(wait>0) std::this_thread::sleep_for(std::chrono::duration<double>(wait));
    now=m_clock.getSeconds();
  }
  slot.cpuMs=m_benchLastTime>0 ? (now-m_benchLastTime)*1000 : 0;
  m_benchLastTime=now;
  if(m_benchTick>=int64_t(m_benchWarmup+m_benchFrames+m_benchSlots.size())) m_app->close();
}

void LodClusters::benchmarkCapture(VkCommandBuffer cmd)
{
  auto& slot=m_benchSlots[m_resources.getFrameCycleIndex()];
  slot.pending=true;
  if(m_renderScene->useStreaming) m_renderScene->sceneStreaming.getStats(slot.stream);
  slot.geometryUsed=m_renderScene->getGeometrySize(false);
  slot.geometryReserved=m_renderScene->getGeometrySize(true);
  slot.operations=m_renderScene->getOperationsSize();
  bool measured=slot.frame>=0 && slot.frame<int64_t(m_benchFrames);
  bool checkpoint=measured && (slot.frame%50==0 || slot.frame==int64_t(m_benchFrames)-1);
  bool recoveryWindow=slot.frame<60 || (slot.frame>=240 && slot.frame<360) || (slot.frame>=480 && slot.frame<600);
  slot.capture=measured && (m_benchCapture==2 || (m_benchCapture==1 && checkpoint) || (m_benchCapture==3 && recoveryWindow));
  if(slot.capture)
  {
    auto& fb=m_resources.m_frameBuffer;
    m_resources.cmdImageTransition(cmd,fb.imgDepthStencil,VK_IMAGE_ASPECT_DEPTH_BIT|VK_IMAGE_ASPECT_STENCIL_BIT,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    m_resources.cmdImageTransition(cmd,fb.imgColor,VK_IMAGE_ASPECT_COLOR_BIT,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    VkBufferImageCopy copy{};
    copy.imageSubresource.aspectMask=VK_IMAGE_ASPECT_DEPTH_BIT;
    copy.imageSubresource.layerCount=1;
    copy.imageExtent={m_windowSize.width,m_windowSize.height,1};
    vkCmdCopyImageToBuffer(cmd,fb.imgDepthStencil.image,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,slot.image.buffer,1,&copy);
    copy.bufferOffset=VkDeviceSize(m_windowSize.width)*m_windowSize.height*4;
    copy.imageSubresource.aspectMask=VK_IMAGE_ASPECT_COLOR_BIT;
    vkCmdCopyImageToBuffer(cmd,fb.imgColor.image,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,slot.image.buffer,1,&copy);
    m_resources.cmdImageTransition(cmd,fb.imgDepthStencil,VK_IMAGE_ASPECT_DEPTH_BIT|VK_IMAGE_ASPECT_STENCIL_BIT,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    m_resources.cmdImageTransition(cmd,fb.imgColor,VK_IMAGE_ASPECT_COLOR_BIT,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;barrier.dstAccessMask=VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&barrier,0,nullptr,0,nullptr);
  }
  ++m_benchTick;
}

void LodClusters::benchmarkConsume(uint32_t index)
{
  auto& slot=m_benchSlots[index];
  if(!slot.pending) return;
  slot.pending=false;
  if(slot.frame<0 || slot.frame>=int64_t(m_benchFrames)) return;
  // The application waited for this frame-cycle fence before entering onRender.
  m_resources.m_allocator.invalidateBuffer(m_resources.m_commonBuffers.readBackHost);
  shaderio::Readback readback=m_resources.m_commonBuffers.readBackHost.data()[index];
  uint64_t q[24]{};
  NVVK_CHECK(vkGetQueryPoolResults(m_app->getDevice(),m_resources.benchmarkQueries,index*24,4,
      4*sizeof(uint64_t),q,sizeof(uint64_t),VK_QUERY_RESULT_64_BIT|VK_QUERY_RESULT_WAIT_BIT));
  if(readback.twoPassCullingActive)
    NVVK_CHECK(vkGetQueryPoolResults(m_app->getDevice(),m_resources.benchmarkQueries,index*24+4,6,
        6*sizeof(uint64_t),q+4,sizeof(uint64_t),VK_QUERY_RESULT_64_BIT|VK_QUERY_RESULT_WAIT_BIT));
  NVVK_CHECK(vkGetQueryPoolResults(m_app->getDevice(),m_resources.benchmarkQueries,index*24+12,4,
      4*sizeof(uint64_t),q+12,sizeof(uint64_t),VK_QUERY_RESULT_64_BIT|VK_QUERY_RESULT_WAIT_BIT));
  NVVK_CHECK(vkGetQueryPoolResults(m_app->getDevice(),m_resources.benchmarkQueries,index*24+20,2,
      2*sizeof(uint64_t),q+20,sizeof(uint64_t),VK_QUERY_RESULT_64_BIT|VK_QUERY_RESULT_WAIT_BIT));
  if(readback.twoPassCullingActive) {
    NVVK_CHECK(vkGetQueryPoolResults(m_app->getDevice(),m_resources.benchmarkQueries,index*24+16,4,
        4*sizeof(uint64_t),q+16,sizeof(uint64_t),VK_QUERY_RESULT_64_BIT|VK_QUERY_RESULT_WAIT_BIT));
    NVVK_CHECK(vkGetQueryPoolResults(m_app->getDevice(),m_resources.benchmarkQueries,index*24+22,2,
        2*sizeof(uint64_t),q+22,sizeof(uint64_t),VK_QUERY_RESULT_64_BIT|VK_QUERY_RESULT_WAIT_BIT));
  }
  auto ms=[&](unsigned a,unsigned b){return double(q[b]-q[a])*m_benchPeriod;};
  double traversal=ms(2,3)+(readback.twoPassCullingActive?ms(4,5):0);
  double second=readback.twoPassCullingActive?ms(6,7)+ms(8,9):0;
  m_benchCsv << slot.frame << ',' << slot.segment << ',' << ms(0,1) << ',' << traversal << ',' << second << ','
    << slot.cpuMs << ',' << readback.twoPassCullingActive << ',' << readback.adaptiveTemporalBypass << ','
    << readback.adaptiveTriggerReason << ',' << readback.numRenderedTriangles << ',' << readback.numRenderedClusters << ','
    << readback.numTraversedTasks << ',' << readback.numRenderedClustersPass0 << ',' << readback.numRenderedClustersPass1 << ','
    << readback.numTraversedTasksPass0 << ',' << readback.numTraversedTasksPass1 << ','
    << readback.numAssemblyRejectedInstances << ',' << readback.numLowDetailInstances << ',' << readback.numQueuedRootInstances << ','
    << readback.adaptiveMotionPixels << ',' << slot.capture << ','
    << (m_rendererConfig.useAssemblyPrefilter?ms(12,13)+(readback.twoPassCullingActive?ms(16,17):0):0) << ','
    << ms(14,15)+(readback.twoPassCullingActive?ms(18,19):0) << ','
    << ms(20,21)+(readback.twoPassCullingActive?ms(22,23):0) << ','
    << readback.numInitializedInstances << ',' << readback.numCompactedCandidates << ',' << readback.numInitializationInvocations << ','
    << slot.stream.geometryUploadBytes << ',' << slot.stream.transferBytes << ',' << slot.stream.loadCount << ',' << slot.stream.unloadCount << ','
    << slot.stream.uncompletedLoadCount << ',' << slot.stream.couldNotTransfer << ',' << slot.stream.couldNotStore << ',' << slot.stream.couldNotAllocateGroup << ','
    << slot.stream.usedDataBytes << ',' << slot.stream.reservedDataBytes << ',' << slot.stream.maxDataBytes << ',' << slot.stream.persistentDataBytes << ','
    << slot.stream.residentGroups << ',' << slot.geometryUsed << ',' << slot.geometryReserved << ',' << slot.operations << ',' << readback.numP1OmittedInstances << '\n';
  if(slot.capture)
  {
    NVVK_CHECK(m_resources.m_allocator.invalidateBuffer(slot.image));
    size_t rawSize=size_t(m_windowSize.width)*m_windowSize.height*8;
    const size_t pixels=size_t(m_windowSize.width)*m_windowSize.height;
    // Lossless byte-plane packing makes the smooth float-depth bytes compress
    // efficiently, keeping all 600 full-resolution frames practical to retain.
    std::vector<uint8_t> planes(rawSize);
    for(size_t channel=0;channel<4;++channel)
      for(size_t pixel=0;pixel<pixels;++pixel)
      {
        planes[channel*pixels+pixel]=slot.image.mapping[pixel*4+channel];
        planes[(channel+4)*pixels+pixel]=slot.image.mapping[pixels*4+pixel*4+channel];
      }
    std::vector<char> compressed(ZSTD_compressBound(rawSize));
    size_t size=ZSTD_compress(compressed.data(),compressed.size(),planes.data(),rawSize,1);
    if(ZSTD_isError(size)) throw std::runtime_error(ZSTD_getErrorName(size));
    auto filename=std::filesystem::path(m_benchOutput)/fmt::format("frame_{:06}.zst",slot.frame);
    std::ofstream file(filename,std::ios::binary);file.write(compressed.data(),std::streamsize(size));
    if(!file) throw std::runtime_error("Benchmark frame write failed");
  }
  if(slot.frame%120==0) {m_benchCsv.flush();LOGI("BENCHMARK_FRAME,%lld\n",slot.frame);}
}

void LodClusters::benchmarkFinish()
{
  if(m_benchSlots.empty()) return;
  for(uint32_t i=0;i<m_benchSlots.size();++i) benchmarkConsume(i);
  m_benchCsv.close();
  for(auto& slot:m_benchSlots) m_resources.m_allocator.destroyBuffer(slot.image);
  vkDestroyQueryPool(m_app->getDevice(),m_resources.benchmarkQueries,nullptr);
  m_resources.benchmarkQueries=VK_NULL_HANDLE;
  m_resources.benchmarkForcedTwoPass=-1;
  m_benchSlots.clear();
  LOGI("BENCHMARK_COMPLETE,%s\n",m_benchOutput.c_str());
}
}


