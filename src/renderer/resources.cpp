//==============================================================================
// src/renderer/resources.cpp
// Implements shared Vulkan resource lifetime and per-frame utility passes.
// Framebuffer rebuilds, Hi-Z construction, temporary command submission, and memory accounting are kept outside renderer-specific code.
//==============================================================================
#include <volk.h>
#include <nvutils/file_operations.hpp>
#include <nvutils/logger.hpp>
#include <nvutils/spirv.hpp>
#include <nvvk/barriers.hpp>
#include <nvvk/formats.hpp>
#include "resources.hpp"


namespace lodclusters {


void Resources::beginFrame(uint32_t cycleIndex)
{
  m_cycleIndex = cycleIndex;
}


void Resources::postProcessFrame(VkCommandBuffer cmd, const FrameConfig& frame, nvvk::ProfilerGpuTimer& profiler)
{

  auto sec = profiler.cmdFrameSection(cmd, "Post-process");

  if(m_frameBuffer.useResolved)
  {
    VkImageBlit region               = {0};
    region.dstOffsets[1].x           = frame.windowSize.width;
    region.dstOffsets[1].y           = frame.windowSize.height;
    region.dstOffsets[1].z           = 1;
    region.srcOffsets[1].x           = m_frameBuffer.targetSize.width;
    region.srcOffsets[1].y           = m_frameBuffer.targetSize.height;
    region.srcOffsets[1].z           = 1;
    region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.dstSubresource.layerCount = 1;
    region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.srcSubresource.layerCount = 1;


    cmdImageTransition(cmd, m_frameBuffer.imgColor, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    cmdImageTransition(cmd, m_frameBuffer.imgColorResolved, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    vkCmdBlitImage(cmd, m_frameBuffer.imgColor.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   m_frameBuffer.imgColorResolved.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region, VK_FILTER_LINEAR);


    cmdImageTransition(cmd, m_frameBuffer.imgColorResolved, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  }
  else
  {

    cmdImageTransition(cmd, m_frameBuffer.imgColor, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  }

  {
    nvvk::cmdMemoryBarrier(cmd, s_supportedShaderStages, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                           VK_ACCESS_2_TRANSFER_READ_BIT);

    VkBufferCopy region;
    region.size      = sizeof(shaderio::Readback);
    region.srcOffset = 0;
    region.dstOffset = m_cycleIndex * sizeof(shaderio::Readback);

    vkCmdCopyBuffer(cmd, m_commonBuffers.readBack.buffer, m_commonBuffers.readBackHost.buffer, 1, &region);
  }
}

void Resources::endFrame() {}


void Resources::emptyFrame(VkCommandBuffer cmd, const FrameConfig& frame, nvvk::ProfilerGpuTimer& profiler)
{

  auto sec = profiler.cmdFrameSection(cmd, "Render");

  cmdBeginRendering(cmd);

  vkCmdEndRendering(cmd);
}


void Resources::trackMemoryUsage(VkDeviceSize size, VmaMemoryUsage usage)
{
  m_memoryUsage.total += size;

  switch(usage)
  {
    case VMA_MEMORY_USAGE_GPU_ONLY:
      m_memoryUsage.deviceLocal += size;
      break;
    case VMA_MEMORY_USAGE_CPU_ONLY:
      m_memoryUsage.hostVisible += size;
      break;
    case VMA_MEMORY_USAGE_CPU_TO_GPU:
    case VMA_MEMORY_USAGE_GPU_TO_CPU:
      m_memoryUsage.hostCached += size;
      break;
    case VMA_MEMORY_USAGE_AUTO:
    case VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE:
    case VMA_MEMORY_USAGE_AUTO_PREFER_HOST:

      break;
  }
}


void Resources::logMemoryUsage() const
{

  LOGI("Memory Usage:");
  LOGI("  Total: %s", formatMemorySize(m_memoryUsage.total).c_str());
  LOGI("  Device Local: %s", formatMemorySize(m_memoryUsage.deviceLocal).c_str());
  LOGI("  Host Visible: %s", formatMemorySize(m_memoryUsage.hostVisible).c_str());
  LOGI("  Host Cached: %s", formatMemorySize(m_memoryUsage.hostCached).c_str());

  if(m_tempCmdBufferPool)
  {
    LOGI("  Memory Pool: %zu blocks (%zu free)",
         m_tempCmdBufferPool->getTotalBlocks(),
         m_tempCmdBufferPool->getFreeBlocks());
  }

  LOGI("  Command Buffers: %u total, %u in use",
       m_cmdBufferCount,
       static_cast<uint32_t>(m_cmdBuffersInUse.size() - std::count(m_cmdBuffersInUse.begin(), m_cmdBuffersInUse.end(), false)));
}


void Resources::init(VkDevice device, VkPhysicalDevice physicalDevice, VkInstance instance, const nvvk::QueueInfo& queue, const nvvk::QueueInfo& queueTransfer)
{
  m_device         = device;
  m_physicalDevice = physicalDevice;
  m_queue          = queue;
  m_queueTransfer  = queueTransfer;


  m_physicalDeviceInfo.init(physicalDevice);

  vkGetPhysicalDeviceMemoryProperties(physicalDevice, &m_memoryProperties);

  m_use16bitDispatch = m_physicalDeviceInfo.properties10.limits.maxComputeWorkGroupCount[0] < (1 << 30);

  m_basicGraphicsState.depthStencilState.depthCompareOp = VK_COMPARE_OP_GREATER;

  {
    VkPhysicalDeviceProperties2 props2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    props2.pNext                       = &m_meshShaderPropsEXT;

    if(m_supportsMeshShaderNV)
    {
      m_meshShaderPropsNV.pNext = props2.pNext;
      props2.pNext              = &m_meshShaderPropsNV;
    }

    vkGetPhysicalDeviceProperties2(physicalDevice, &props2);
  }

  {
    VmaAllocatorCreateInfo allocatorInfo = {
        .flags          = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
        .physicalDevice = physicalDevice,
        .device         = device,
        .instance       = instance,
    };

    NVVK_CHECK(m_allocator.init(allocatorInfo));


  }


  m_uploader.init(&m_allocator);


  m_samplerPool.init(device);

  m_samplerPool.acquireSampler(m_samplerLinear);


  {
    VkCommandPoolCreateInfo createInfo = {
        .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = m_queue.familyIndex,
    };

    NVVK_CHECK(vkCreateCommandPool(m_device, &createInfo, nullptr, &m_tempCommandPool));


    m_cmdBufferCount = 32;

    m_cmdBuffers.resize(m_cmdBufferCount);

    m_cmdBuffersInUse.resize(m_cmdBufferCount, false);

    VkCommandBufferAllocateInfo allocInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = m_tempCommandPool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = m_cmdBufferCount,
    };

    NVVK_CHECK(vkAllocateCommandBuffers(m_device, &allocInfo, m_cmdBuffers.data()));
  }


  m_tempCmdBufferPool = std::make_unique<MemoryPool>(4096, 32);

  {

    std::filesystem::path                    exeDirectoryPath = nvutils::getExecutablePath().parent_path();
    const std::vector<std::filesystem::path> searchPaths      = {

        std::filesystem::absolute(exeDirectoryPath / TARGET_EXE_TO_SOURCE_DIRECTORY / "shaders"),
        std::filesystem::absolute(exeDirectoryPath / TARGET_EXE_TO_SOURCE_DIRECTORY / "shaders" / "interface"),
        std::filesystem::absolute(exeDirectoryPath / TARGET_EXE_TO_SOURCE_DIRECTORY / "shaders" / "common"),
        std::filesystem::absolute(exeDirectoryPath / TARGET_EXE_TO_SOURCE_DIRECTORY / "shaders" / "render"),
        std::filesystem::absolute(exeDirectoryPath / TARGET_EXE_TO_SOURCE_DIRECTORY / "shaders" / "debug"),
        std::filesystem::absolute(exeDirectoryPath / TARGET_EXE_TO_SOURCE_DIRECTORY / "shaders" / "post"),
        std::filesystem::absolute(exeDirectoryPath / TARGET_EXE_TO_SOURCE_DIRECTORY / "shaders" / "streaming"),
        std::filesystem::absolute(exeDirectoryPath / TARGET_EXE_TO_SOURCE_DIRECTORY / "shaders" / "traversal"),
        std::filesystem::absolute(exeDirectoryPath / TARGET_EXE_TO_SOURCE_DIRECTORY / "shaders" / "build"),
        std::filesystem::absolute(exeDirectoryPath / TARGET_EXE_TO_NVSHADERS_DIRECTORY),

        std::filesystem::absolute(exeDirectoryPath / TARGET_NAME "_files" / "shaders"),
        std::filesystem::absolute(exeDirectoryPath / TARGET_NAME "_files" / "shaders" / "interface"),
        std::filesystem::absolute(exeDirectoryPath / TARGET_NAME "_files" / "shaders" / "common"),
        std::filesystem::absolute(exeDirectoryPath / TARGET_NAME "_files" / "shaders" / "render"),
        std::filesystem::absolute(exeDirectoryPath / TARGET_NAME "_files" / "shaders" / "debug"),
        std::filesystem::absolute(exeDirectoryPath / TARGET_NAME "_files" / "shaders" / "post"),
        std::filesystem::absolute(exeDirectoryPath / TARGET_NAME "_files" / "shaders" / "streaming"),
        std::filesystem::absolute(exeDirectoryPath / TARGET_NAME "_files" / "shaders" / "traversal"),
        std::filesystem::absolute(exeDirectoryPath / TARGET_NAME "_files" / "shaders" / "build"),
        std::filesystem::absolute(exeDirectoryPath),
    };

    m_glslCompiler.addSearchPaths(searchPaths);

    m_glslCompiler.defaultOptions();

    m_glslCompiler.defaultTarget();

    m_glslCompiler.options().SetGenerateDebugInfo();
  }


  {
    m_allocator.createBuffer(m_commonBuffers.frameConstants, sizeof(shaderio::FrameConstants),
                             VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

    m_allocator.createBuffer(m_commonBuffers.readBack, sizeof(shaderio::Readback),
                             VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT,
                             VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    m_allocator.createBuffer(m_commonBuffers.readBackHost, sizeof(shaderio::Readback) * 4,
                             VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_ONLY,
                             VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);
  }

  {
    NVHizVK::Config config;
    config.msaaSamples             = 0;
    config.reversedZ               = true;
    config.supportsMinmaxFilter    = true;
    config.supportsSubGroupShuffle = true;


    m_hiz.init(m_device, config, 2);

    shaderc::SpvCompilationResult shaderResults[NVHizVK::SHADER_COUNT];
    for(uint32_t i = 0; i < NVHizVK::SHADER_COUNT; i++)
    {

      shaderc::CompileOptions options = makeCompilerOptions();

      m_hiz.appendShaderDefines(i, options);

      compileShader(shaderResults[i], VK_SHADER_STAGE_COMPUTE_BIT, "post/hiz.comp.glsl", &options);
    }

    m_hiz.initPipelines(shaderResults);
  }
  {
    VrdxSorterCreateInfo sorterCreateInfo;
    sorterCreateInfo.device         = m_device;
    sorterCreateInfo.physicalDevice = m_physicalDevice;
    sorterCreateInfo.pipelineCache  = nullptr;


    vrdxCreateSorter(&sorterCreateInfo, &m_vrdxSorter);
  }
  {

    m_queueStates.primary.init(m_device, m_queue.queue, m_queue.familyIndex, 0);

    NVVK_DBG_NAME(m_queueStates.primary.m_timelineSemaphore);


    m_queueStates.transfer.init(m_device, m_queueTransfer.queue, m_queueTransfer.familyIndex, 0);

    NVVK_DBG_NAME(m_queueStates.transfer.m_timelineSemaphore);
  }
}


void Resources::deinit()
{
  NVVK_CHECK(vkDeviceWaitIdle(m_device));


  m_allocator.destroyBuffer(m_commonBuffers.frameConstants);

  m_allocator.destroyBuffer(m_commonBuffers.readBack);

  m_allocator.destroyBuffer(m_commonBuffers.readBackHost);


  if(!m_cmdBuffers.empty())
  {
    vkFreeCommandBuffers(m_device, m_tempCommandPool, m_cmdBufferCount, m_cmdBuffers.data());
  }


  vkDestroyCommandPool(m_device, m_tempCommandPool, nullptr);

  deinitFramebuffer();

  m_hiz.deinit();

  vrdxDestroySorter(m_vrdxSorter);

  m_queueStates.primary.deinit();

  m_queueStates.transfer.deinit();


  m_samplerPool.releaseSampler(m_samplerLinear);

  m_samplerPool.deinit();

  m_uploader.deinit();

  m_allocator.deinit();
}


bool Resources::initFramebuffer(const VkExtent2D& windowSize, int supersample)
{

  bool needsRecreation = false;
  bool wasResize = false;

  if(m_frameBuffer.imgColor.image != 0)
  {

    bool sizeChanged = (m_frameBuffer.targetSize.width != windowSize.width * std::min(supersample, 4) ||
                        m_frameBuffer.targetSize.height != windowSize.height * std::min(supersample, 4));
    bool supersampleChanged = (m_frameBuffer.supersample != supersample);

    if(sizeChanged || supersampleChanged)
    {

      deinitFramebuffer();
      wasResize = true;
      needsRecreation = true;
    }
  }
  else
  {
    needsRecreation = true;
  }

  if(!needsRecreation)
  {

    m_frameBuffer.windowSize = windowSize;
    return true;
  }

  m_fboChangeID++;

  bool oldResolved = m_frameBuffer.supersample > 1;

  switch(supersample)
  {
    case 720:
      m_frameBuffer.targetSize.width  = 1280;
      m_frameBuffer.targetSize.height = 720;
      break;
    case 1080:
      m_frameBuffer.targetSize.width  = 1920;
      m_frameBuffer.targetSize.height = 1080;
      break;
    case 1440:
      m_frameBuffer.targetSize.width  = 2560;
      m_frameBuffer.targetSize.height = 1440;
      break;
    case 2160:
      m_frameBuffer.targetSize.width  = 3840;
      m_frameBuffer.targetSize.height = 2160;
      break;
    case 1024:
      m_frameBuffer.targetSize.width  = 1024;
      m_frameBuffer.targetSize.height = 1024;
      break;
    case 2048:
      m_frameBuffer.targetSize.width  = 2048;
      m_frameBuffer.targetSize.height = 2048;
      break;
    case 4096:
      m_frameBuffer.targetSize.width  = 4096;
      m_frameBuffer.targetSize.height = 4096;
      break;
    default:

      m_frameBuffer.targetSize.width  = windowSize.width * std::min(supersample, 4);

      m_frameBuffer.targetSize.height = windowSize.height * std::min(supersample, 4);
      break;
  }

  m_frameBuffer.pixelScale = std::max(float(m_frameBuffer.targetSize.width) / float(windowSize.width),
                                      float(m_frameBuffer.targetSize.height) / float(windowSize.height));
  m_basicGraphicsState.rasterizationState.lineWidth = m_frameBuffer.pixelScale;
  m_frameBuffer.renderSize = m_frameBuffer.targetSize;
  m_frameBuffer.windowSize = windowSize;

  m_frameBuffer.supersample = supersample;
  LOGI("framebuffer: %d x %d (target)\n", m_frameBuffer.targetSize.width, m_frameBuffer.targetSize.height);

  m_frameBuffer.useResolved = supersample > 1;

  VkSampleCountFlagBits samplesUsed = VK_SAMPLE_COUNT_1_BIT;
  {

    VkImageCreateInfo cbImageInfo = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    cbImageInfo.imageType         = VK_IMAGE_TYPE_2D;
    cbImageInfo.format            = m_frameBuffer.colorFormat;
    cbImageInfo.extent.width      = m_frameBuffer.targetSize.width;
    cbImageInfo.extent.height     = m_frameBuffer.targetSize.height;
    cbImageInfo.extent.depth      = 1;
    cbImageInfo.mipLevels         = 1;
    cbImageInfo.arrayLayers       = 1;
    cbImageInfo.samples           = samplesUsed;
    cbImageInfo.tiling            = VK_IMAGE_TILING_OPTIMAL;
    cbImageInfo.flags             = 0;
    cbImageInfo.initialLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
    cbImageInfo.usage             = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
                        | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    VkImageViewCreateInfo cbImageViewInfo           = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    cbImageViewInfo.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
    cbImageViewInfo.format                          = m_frameBuffer.colorFormat;
    cbImageViewInfo.components.r                    = VK_COMPONENT_SWIZZLE_R;
    cbImageViewInfo.components.g                    = VK_COMPONENT_SWIZZLE_G;
    cbImageViewInfo.components.b                    = VK_COMPONENT_SWIZZLE_B;
    cbImageViewInfo.components.a                    = VK_COMPONENT_SWIZZLE_A;
    cbImageViewInfo.flags                           = 0;
    cbImageViewInfo.subresourceRange.levelCount     = 1;
    cbImageViewInfo.subresourceRange.baseMipLevel   = 0;
    cbImageViewInfo.subresourceRange.layerCount     = 1;
    cbImageViewInfo.subresourceRange.baseArrayLayer = 0;
    cbImageViewInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;

    NVVK_CHECK(m_allocator.createImage(m_frameBuffer.imgColor, cbImageInfo, cbImageViewInfo));

    NVVK_DBG_NAME(m_frameBuffer.imgColor.image);

    NVVK_DBG_NAME(m_frameBuffer.imgColor.descriptor.imageView);
  }

  if(m_frameBuffer.useResolved)
  {

    VkImageCreateInfo resImageInfo = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    resImageInfo.imageType         = VK_IMAGE_TYPE_2D;
    resImageInfo.format            = m_frameBuffer.colorFormat;
    resImageInfo.extent.width      = windowSize.width;
    resImageInfo.extent.height     = windowSize.height;
    resImageInfo.extent.depth      = 1;
    resImageInfo.mipLevels         = 1;
    resImageInfo.arrayLayers       = 1;
    resImageInfo.samples           = VK_SAMPLE_COUNT_1_BIT;
    resImageInfo.tiling            = VK_IMAGE_TILING_OPTIMAL;
    resImageInfo.usage             = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
                         | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    resImageInfo.flags         = 0;
    resImageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkImageViewCreateInfo resImageViewInfo           = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    resImageViewInfo.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
    resImageViewInfo.format                          = m_frameBuffer.colorFormat;
    resImageViewInfo.components.r                    = VK_COMPONENT_SWIZZLE_R;
    resImageViewInfo.components.g                    = VK_COMPONENT_SWIZZLE_G;
    resImageViewInfo.components.b                    = VK_COMPONENT_SWIZZLE_B;
    resImageViewInfo.components.a                    = VK_COMPONENT_SWIZZLE_A;
    resImageViewInfo.flags                           = 0;
    resImageViewInfo.subresourceRange.levelCount     = 1;
    resImageViewInfo.subresourceRange.baseMipLevel   = 0;
    resImageViewInfo.subresourceRange.layerCount     = 1;
    resImageViewInfo.subresourceRange.baseArrayLayer = 0;
    resImageViewInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;

    NVVK_CHECK(m_allocator.createImage(m_frameBuffer.imgColorResolved, resImageInfo, resImageViewInfo));

    NVVK_DBG_NAME(m_frameBuffer.imgColorResolved.image);

    NVVK_DBG_NAME(m_frameBuffer.imgColorResolved.descriptor.imageView);
  }


  {

    VkCommandBuffer cmd = createTempCmdBuffer();

    updateFramebufferRenderSizeDependent(cmd);

    tempSyncSubmit(cmd);
  }


  {
    VkPipelineRenderingCreateInfo pipelineRenderingInfo = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};

    pipelineRenderingInfo.colorAttachmentCount    = 1;
    pipelineRenderingInfo.pColorAttachmentFormats = &m_frameBuffer.colorFormat;
    pipelineRenderingInfo.depthAttachmentFormat   = m_frameBuffer.depthStencilFormat;

    m_frameBuffer.pipelineRenderingInfo = pipelineRenderingInfo;
  }

  return true;
}


void Resources::updateFramebufferRenderSizeDependent(VkCommandBuffer cmd)
{
  VkSampleCountFlagBits samplesUsed = VK_SAMPLE_COUNT_1_BIT;


  // Reversed Z needs floating-point depth to retain precision far from the near
  // plane. The generic helper prefers D24, which quantizes close CAD surfaces
  // to the same depth and makes their winner depend on parallel draw order.
  m_frameBuffer.depthStencilFormat = preferFloatDepth
      ? nvvk::findSupportedFormat(m_physicalDevice,
          {VK_FORMAT_D32_SFLOAT_S8_UINT,VK_FORMAT_D24_UNORM_S8_UINT,VK_FORMAT_D16_UNORM_S8_UINT},
          VK_IMAGE_TILING_OPTIMAL,VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)
      : nvvk::findDepthStencilFormat(m_physicalDevice);
  LOGI("Depth format: %s (reversed Z)\n",m_frameBuffer.depthStencilFormat==VK_FORMAT_D32_SFLOAT_S8_UINT
      ? "D32_SFLOAT_S8_UINT" : "fixed-point fallback");

  {
    VkImageCreateInfo dsImageInfo = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    dsImageInfo.imageType         = VK_IMAGE_TYPE_2D;
    dsImageInfo.format            = m_frameBuffer.depthStencilFormat;
    dsImageInfo.extent.width      = m_frameBuffer.renderSize.width;
    dsImageInfo.extent.height     = m_frameBuffer.renderSize.height;
    dsImageInfo.extent.depth      = 1;
    dsImageInfo.mipLevels         = 1;
    dsImageInfo.arrayLayers       = 1;
    dsImageInfo.samples           = samplesUsed;
    dsImageInfo.tiling            = VK_IMAGE_TILING_OPTIMAL;
    dsImageInfo.usage             = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    dsImageInfo.flags             = 0;
    dsImageInfo.initialLayout     = VK_IMAGE_LAYOUT_UNDEFINED;

    VkImageViewCreateInfo dsImageViewInfo           = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    dsImageViewInfo.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
    dsImageViewInfo.format                          = m_frameBuffer.depthStencilFormat;
    dsImageViewInfo.components.r                    = VK_COMPONENT_SWIZZLE_R;
    dsImageViewInfo.components.g                    = VK_COMPONENT_SWIZZLE_G;
    dsImageViewInfo.components.b                    = VK_COMPONENT_SWIZZLE_B;
    dsImageViewInfo.components.a                    = VK_COMPONENT_SWIZZLE_A;
    dsImageViewInfo.flags                           = 0;
    dsImageViewInfo.subresourceRange.levelCount     = 1;
    dsImageViewInfo.subresourceRange.baseMipLevel   = 0;
    dsImageViewInfo.subresourceRange.layerCount     = 1;
    dsImageViewInfo.subresourceRange.baseArrayLayer = 0;
    dsImageViewInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_STENCIL_BIT | VK_IMAGE_ASPECT_DEPTH_BIT;

    NVVK_CHECK(m_allocator.createImage(m_frameBuffer.imgDepthStencil, dsImageInfo, dsImageViewInfo));

    NVVK_DBG_NAME(m_frameBuffer.imgDepthStencil.image);

    NVVK_DBG_NAME(m_frameBuffer.imgDepthStencil.descriptor.imageView);

    dsImageViewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    dsImageViewInfo.image                       = m_frameBuffer.imgDepthStencil.image;

    NVVK_CHECK(vkCreateImageView(m_device, &dsImageViewInfo, nullptr, &m_frameBuffer.viewDepth));
  }

  for(uint32_t i = 0; i < 2; i++)
  {

    m_hiz.setupUpdateInfos(m_hizUpdate[i], m_frameBuffer.renderSize.width, m_frameBuffer.renderSize.height,
                           m_frameBuffer.depthStencilFormat, VK_IMAGE_ASPECT_DEPTH_BIT);


    bool needsRecreate = false;
    if(m_frameBuffer.imgHizFar[i].image == VK_NULL_HANDLE) {
      needsRecreate = true;
    } else {


      needsRecreate = true;
    }

    if(needsRecreate) {

      VkImageCreateInfo hizImageInfo = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
      hizImageInfo.imageType         = VK_IMAGE_TYPE_2D;
      hizImageInfo.format            = m_hizUpdate[i].farInfo.format;
      hizImageInfo.extent.width      = m_hizUpdate[i].farInfo.width;
      hizImageInfo.extent.height     = m_hizUpdate[i].farInfo.height;
      hizImageInfo.mipLevels         = m_hizUpdate[i].farInfo.mipLevels;
      hizImageInfo.extent.depth      = 1;
      hizImageInfo.arrayLayers       = 1;
      hizImageInfo.samples           = VK_SAMPLE_COUNT_1_BIT;
      hizImageInfo.tiling            = VK_IMAGE_TILING_OPTIMAL;
      hizImageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
      hizImageInfo.flags = 0;
      hizImageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;


      m_allocator.destroyImage(m_frameBuffer.imgHizFar[i]);
      NVVK_CHECK(m_allocator.createImage(m_frameBuffer.imgHizFar[i], hizImageInfo));

      NVVK_DBG_NAME(m_frameBuffer.imgHizFar[i].image);
    }


    m_hizUpdate[i].sourceImage = m_frameBuffer.imgDepthStencil.image;
    m_hizUpdate[i].farImage    = m_frameBuffer.imgHizFar[i].image;
    m_hizUpdate[i].nearImage   = VK_NULL_HANDLE;


    m_hiz.deinitUpdateViews(m_hizUpdate[i]);

    m_hiz.initUpdateViews(m_hizUpdate[i]);

    m_hiz.updateDescriptorSet(m_hizUpdate[i], i);
  }


  cmdImageTransition(cmd, m_frameBuffer.imgHizFar[0], VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);

  cmdImageTransition(cmd, m_frameBuffer.imgHizFar[1], VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);
  {
    VkClearColorValue clear = {};
    clear.float32[0]        = 0.0f;
    clear.float32[1]        = 0.0f;
    clear.float32[2]        = 0.0f;
    clear.float32[3]        = 0.0f;

    VkImageSubresourceRange subResourceRange;
    subResourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    subResourceRange.baseArrayLayer = 0;
    subResourceRange.baseMipLevel   = 0;
    subResourceRange.layerCount     = VK_REMAINING_ARRAY_LAYERS;
    subResourceRange.levelCount     = VK_REMAINING_MIP_LEVELS;

    vkCmdClearColorImage(cmd, m_frameBuffer.imgHizFar[0].image, VK_IMAGE_LAYOUT_GENERAL, &clear, 1, &subResourceRange);

    vkCmdClearColorImage(cmd, m_frameBuffer.imgHizFar[1].image, VK_IMAGE_LAYOUT_GENERAL, &clear, 1, &subResourceRange);
  }

  {
    VkViewport vp;
    VkRect2D   sc;
    vp.x        = 0;
    vp.y        = 0;

    vp.width    = float(m_frameBuffer.renderSize.width);

    vp.height   = float(m_frameBuffer.renderSize.height);
    vp.minDepth = 0.0f;
    vp.maxDepth = 1.0f;

    sc.offset.x = 0;
    sc.offset.y = 0;
    sc.extent   = m_frameBuffer.renderSize;

    m_frameBuffer.viewport = vp;
    m_frameBuffer.scissor  = sc;
  }
}


void Resources::deinitFramebufferRenderSizeDependent()
{


  m_allocator.destroyImage(m_frameBuffer.imgDepthStencil);


  m_allocator.destroyImage(m_frameBuffer.imgHizFar[0]);

  m_allocator.destroyImage(m_frameBuffer.imgHizFar[1]);
  vkDestroyImageView(m_device, m_frameBuffer.viewDepth, nullptr);
  m_frameBuffer.viewDepth = VK_NULL_HANDLE;


  m_hiz.deinitUpdateViews(m_hizUpdate[0]);

  m_hiz.deinitUpdateViews(m_hizUpdate[1]);
}


void Resources::deinitFramebuffer()
{
  NVVK_CHECK(vkDeviceWaitIdle(m_device));


  m_allocator.destroyImage(m_frameBuffer.imgColor);

  m_allocator.destroyImage(m_frameBuffer.imgColorResolved);


  deinitFramebufferRenderSizeDependent();
}


glm::vec2 Resources::getFramebufferWindow2RenderScale() const
{
  if(m_frameBuffer.supersample >= 720)
  {

    return glm::vec2(1, 1);
  }

  return glm::vec2(m_frameBuffer.renderSize.width, m_frameBuffer.renderSize.height)

         / glm::vec2(m_frameBuffer.windowSize.width, m_frameBuffer.windowSize.height);
}


void Resources::getReadbackData(shaderio::Readback& readback)
{

  const shaderio::Readback* pReadback = m_commonBuffers.readBackHost.data();
  readback                            = pReadback[m_cycleIndex];
}


void Resources::cmdBuildHiz(VkCommandBuffer cmd, const FrameConfig& frame, nvvk::ProfilerGpuTimer& profiler, uint32_t idx)
{

  auto timerSection = profiler.cmdFrameSection(cmd, "HiZ");


  // A single layout is tracked for this depth/stencil resource. Transition
  // both aspects together so sampled descriptors and the next render agree.
  cmdImageTransition(cmd, m_frameBuffer.imgDepthStencil, VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);


  m_hiz.cmdUpdateHiz(cmd, m_hizUpdate[idx], idx);


  cmdImageTransition(cmd, m_frameBuffer.imgHizFar[idx], VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);
}

bool Resources::compileShader(shaderc::SpvCompilationResult& compiled,
                              VkShaderStageFlagBits          shaderStage,
                              const std::filesystem::path&   filePath,
                              shaderc::CompileOptions*       options)
{
  compiled = m_glslCompiler.compileFile(filePath, nvvkglsl::getShaderKind(shaderStage), options);
  if(compiled.GetCompilationStatus() == shaderc_compilation_status_success)
  {
    if(m_dumpSpirv)
    {


      std::filesystem::path dumpFile = filePath.filename();

      dumpFile.replace_extension("spirv");

      nvutils::dumpSpirv(dumpFile, nvvkglsl::GlslCompiler::getSpirv(compiled), nvvkglsl::GlslCompiler::getSpirvSize(compiled));
    }
    return true;
  }
  else
  {

    std::string errorMessage = compiled.GetErrorMessage();
    if(!errorMessage.empty())
      nvutils::Logger::getInstance().log(nvutils::Logger::LogLevel::eWARNING, "%s", errorMessage.c_str());
    return false;
  }
}


VkCommandBuffer Resources::createTempCmdBuffer()
{

  for(uint32_t i = 0; i < m_cmdBufferCount; i++)
  {
    if(!m_cmdBuffersInUse[i])
    {
      m_cmdBuffersInUse[i] = true;
      VkCommandBuffer cmd = m_cmdBuffers[i];

      VkCommandBufferBeginInfo beginInfo = {};
      beginInfo.sType                    = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
      beginInfo.flags                    = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
      beginInfo.pInheritanceInfo         = nullptr;

      NVVK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));
      return cmd;
    }
  }


  uint32_t newCount = m_cmdBufferCount * 2;


  std::vector<VkCommandBuffer> newCmdBuffers(newCount);


  std::vector<bool> newCmdBuffersInUse(newCount, false);


  std::copy(m_cmdBuffers.begin(), m_cmdBuffers.end(), newCmdBuffers.begin());
  std::copy(m_cmdBuffersInUse.begin(), m_cmdBuffersInUse.end(), newCmdBuffersInUse.begin());


  VkCommandBufferAllocateInfo allocInfo = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = m_tempCommandPool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = newCount - m_cmdBufferCount,
  };

  NVVK_CHECK(vkAllocateCommandBuffers(m_device, &allocInfo, newCmdBuffers.data() + m_cmdBufferCount));


  m_cmdBuffers = std::move(newCmdBuffers);

  m_cmdBuffersInUse = std::move(newCmdBuffersInUse);
  m_cmdBufferCount = newCount;


  m_cmdBuffersInUse[m_cmdBufferCount - (newCount - m_cmdBufferCount)] = true;
  VkCommandBuffer cmd = m_cmdBuffers[m_cmdBufferCount - (newCount - m_cmdBufferCount)];

  VkCommandBufferBeginInfo beginInfo = {};
  beginInfo.sType                    = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags                    = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  beginInfo.pInheritanceInfo         = nullptr;

  NVVK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));
  return cmd;
}


void Resources::tempSyncSubmit(VkCommandBuffer cmd)
{

  vkEndCommandBuffer(cmd);

  VkCommandBufferSubmitInfo cmdInfo = {
      .sType         = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
      .commandBuffer = cmd,
  };

  VkSubmitInfo2 submitInfo2 = {
      .sType                  = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
      .flags                  = 0,
      .commandBufferInfoCount = 1,
      .pCommandBufferInfos    = &cmdInfo,
  };

  NVVK_CHECK(vkQueueSubmit2(m_queue.queue, 1, &submitInfo2, nullptr));
  NVVK_CHECK(vkDeviceWaitIdle(m_device));


  for(uint32_t i = 0; i < m_cmdBufferCount; i++)
  {
    if(m_cmdBuffers[i] == cmd)
    {
      m_cmdBuffersInUse[i] = false;
      break;
    }
  }
}


void Resources::cmdBeginRendering(VkCommandBuffer cmd, bool hasSecondary, VkAttachmentLoadOp loadOpColor, VkAttachmentLoadOp loadOpDepth)
{
  VkClearValue colorClear{.color = {m_bgColor.x, m_bgColor.y, m_bgColor.z, m_bgColor.w}};
  VkClearValue depthClear{.depthStencil = {0.0F, 0}};


  cmdImageTransition(cmd, m_frameBuffer.imgColor, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
  cmdImageTransition(cmd, m_frameBuffer.imgDepthStencil, VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT,
                     VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

  VkRenderingAttachmentInfo colorAttachment = {
      .sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView   = m_frameBuffer.imgColor.descriptor.imageView,
      .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .loadOp      = loadOpColor,
      .storeOp     = VK_ATTACHMENT_STORE_OP_STORE,
      .clearValue  = colorClear,
  };


  VkRenderingAttachmentInfo depthStencilAttachment{
      .sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView   = m_frameBuffer.imgDepthStencil.descriptor.imageView,
      .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
      .loadOp      = loadOpDepth,
      .storeOp     = VK_ATTACHMENT_STORE_OP_STORE,
      .clearValue  = depthClear,
  };


  VkRenderingInfo renderingInfo{
      .sType                = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .flags                = hasSecondary ? VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT : VkRenderingFlags(0),
      .renderArea           = m_frameBuffer.scissor,
      .layerCount           = 1,
      .colorAttachmentCount = 1,
      .pColorAttachments    = &colorAttachment,
      .pDepthAttachment     = &depthStencilAttachment,
  };


  vkCmdBeginRendering(cmd, &renderingInfo);


  vkCmdSetViewportWithCount(cmd, 1, &m_frameBuffer.viewport);

  vkCmdSetScissorWithCount(cmd, 1, &m_frameBuffer.scissor);
}


void Resources::cmdImageTransition(VkCommandBuffer cmd, nvvk::Image& rimg, VkImageAspectFlags aspects, VkImageLayout newLayout, bool needBarrier) const
{
  if(newLayout == rimg.descriptor.imageLayout && !needBarrier)
    return;
  nvvk::ImageMemoryBarrierParams imageBarrier;
  imageBarrier.image                       = rimg.image;
  imageBarrier.oldLayout                   = rimg.descriptor.imageLayout;
  imageBarrier.newLayout                   = newLayout;
  imageBarrier.subresourceRange.aspectMask = aspects;


  nvvk::cmdImageMemoryBarrier(cmd, imageBarrier);

  rimg.descriptor.imageLayout = newLayout;
}


VkDeviceSize Resources::getDeviceLocalHeapSize() const
{
  const VkPhysicalDeviceMemoryProperties& memProperties = m_memoryProperties;

  for(uint32_t type = 0; type < memProperties.memoryTypeCount; type++)
  {

    if(memProperties.memoryTypes[type].propertyFlags == VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
    {
      return memProperties.memoryHeaps[memProperties.memoryTypes[type].heapIndex].size;
    }
  }


  for(uint32_t type = 0; type < memProperties.memoryTypeCount; type++)
  {
    if((memProperties.memoryTypes[type].propertyFlags & (VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
       == (VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
    {
      return memProperties.memoryHeaps[memProperties.memoryTypes[type].heapIndex].size;
    }
  }

  assert(0);
  return 0;
}


bool Resources::isBufferSizeValid(VkDeviceSize size) const
{
  return size <= m_physicalDeviceInfo.properties13.maxBufferSize && size <= m_physicalDeviceInfo.properties11.maxMemoryAllocationSize;
}


void QueueState::init(VkDevice device, VkQueue queue, uint32_t familyIndex, uint64_t initialValue)
{

  assert(m_device == nullptr);

  m_device      = device;
  m_queue       = queue;
  m_familyIndex = familyIndex;

  VkSemaphoreTypeCreateInfo timelineSemaphoreCreateInfo{.sType         = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
                                                        .pNext         = nullptr,
                                                        .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
                                                        .initialValue  = initialValue};
  VkSemaphoreCreateInfo     semaphoreCreateInfo{
          .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &timelineSemaphoreCreateInfo, .flags = 0};


  vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &m_timelineSemaphore);

  m_device        = device;
  m_timelineValue = initialValue + 1;
}


void QueueState::deinit()
{
  if(!m_device)
    return;

  vkDestroySemaphore(m_device, m_timelineSemaphore, nullptr);
}


VkSemaphoreSubmitInfo QueueState::getWaitSubmit(VkPipelineStageFlags2 stageMask, uint32_t deviceIndex ) const
{
  VkSemaphoreSubmitInfo signalSubmitInfo{.sType       = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                                         .pNext       = nullptr,
                                         .semaphore   = m_timelineSemaphore,
                                         .value       = m_timelineValue,
                                         .stageMask   = stageMask,
                                         .deviceIndex = deviceIndex};

  return signalSubmitInfo;
}


VkSemaphoreSubmitInfo QueueState::advanceSignalSubmit(VkPipelineStageFlags2 stageMask, uint32_t deviceIndex )
{
  VkSemaphoreSubmitInfo signalSubmitInfo{.sType       = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                                         .pNext       = nullptr,
                                         .semaphore   = m_timelineSemaphore,
                                         .value       = m_timelineValue++,
                                         .stageMask   = stageMask,
                                         .deviceIndex = deviceIndex};

  return signalSubmitInfo;
}
}
