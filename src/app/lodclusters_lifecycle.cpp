//==============================================================================
// src/app/lodclusters_lifecycle.cpp
// Handles application attach/detach, resize, renderer creation, and image descriptor refresh.
// Resource lifetime is staged so framebuffer-dependent objects can be rebuilt without tearing down the full scene.
//==============================================================================



#include <volk.h>
#include <fmt/format.h>
#include <nvutils/file_operations.hpp>
#include "lodclusters.hpp"


namespace lodclusters {


void LodClusters::onResize(VkCommandBuffer cmd, const VkExtent2D& size)
{
  m_windowSize = size;

  m_resources.initFramebuffer(m_windowSize, m_tweak.supersample);

  updateImguiImage();
  if(m_renderer)
  {

    m_renderer->updatedFrameBuffer(m_resources, *m_renderScene);
    m_rendererFboChangeID = m_resources.m_fboChangeID;
  }
}


void LodClusters::updateImguiImage()
{
  if(m_imguiTexture)
  {

    ImGui_ImplVulkan_RemoveTexture(m_imguiTexture);
    m_imguiTexture = nullptr;
  }

  VkImageView imageView = m_resources.m_frameBuffer.useResolved ? m_resources.m_frameBuffer.imgColorResolved.descriptor.imageView :
                                                                  m_resources.m_frameBuffer.imgColor.descriptor.imageView;


  assert(imageView);


  m_imguiTexture = ImGui_ImplVulkan_AddTexture(m_imguiSampler, imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}


void LodClusters::deinitRenderer()
{
  NVVK_CHECK(vkDeviceWaitIdle(m_app->getDevice()));

  if(m_renderer)
  {

    m_renderer->deinit(m_resources);
    m_renderer = nullptr;
  }
}


void LodClusters::initRenderer(RendererType rtype)
{

  LOGI("Initializing renderer and compiling shaders\n");

  deinitRenderer();
  if(!m_renderScene)
    return;


  printf("init renderer %d\n", rtype);

  if(m_renderScene->useStreaming)
  {
    if(!m_renderScene->sceneStreaming.reloadShaders())
    {

      LOGE("RenderScene shaders failed\n");
      return;
    }
  }

  switch(rtype)
  {
    case RENDERER_RASTER_CLUSTERS_LOD:

      m_renderer = makeRendererRasterClustersLod();
      break;
  }

  if(m_renderer && !m_renderer->init(m_resources, *m_renderScene, m_rendererConfig))
  {
    m_renderer = nullptr;

    LOGE("Renderer init failed\n");
  }

  m_rendererFboChangeID = m_resources.m_fboChangeID;
}


void LodClusters::onAttach(nvapp::Application* app)
{
  m_app = app;


  m_tweak.supersample = std::max(1, m_tweak.supersample);

  m_info.cameraManipulator->setMode(nvutils::CameraManipulator::Fly);
  m_renderer = nullptr;

  if(m_resources.m_supportsSmBuiltinsNV)
  {
    VkPhysicalDeviceProperties2 physicalProperties = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    VkPhysicalDeviceShaderSMBuiltinsPropertiesNV smProperties = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SM_BUILTINS_PROPERTIES_NV};
    physicalProperties.pNext = &smProperties;
    vkGetPhysicalDeviceProperties2(app->getPhysicalDevice(), &physicalProperties);


    if(smProperties.shaderSMCount * smProperties.shaderWarpsPerSM > 4096)
      m_frameConfig.traversalPersistentThreads = smProperties.shaderSMCount * smProperties.shaderWarpsPerSM * 2;
    else if(smProperties.shaderSMCount * smProperties.shaderWarpsPerSM > 2048 + 1024)
      m_frameConfig.traversalPersistentThreads = smProperties.shaderSMCount * smProperties.shaderWarpsPerSM * 4;
    else
      m_frameConfig.traversalPersistentThreads = smProperties.shaderSMCount * smProperties.shaderWarpsPerSM * 8;
  }

  {

    m_ui.enumAdd(GUI_RENDERER, RENDERER_RASTER_CLUSTERS_LOD, "Rasterization");

    {
      for(uint32_t i = 0; i < NUM_CLUSTER_CONFIGS; i++)
      {
        std::string enumStr = fmt::format("{}T_{}V", s_clusterInfos[i].tris, s_clusterInfos[i].verts);
        m_ui.enumAdd(GUI_MESHLET, s_clusterInfos[i].cfg, enumStr.c_str());
      }
    }


    m_ui.enumAdd(GUI_SUPERSAMPLE, 1, "none");

    m_ui.enumAdd(GUI_SUPERSAMPLE, 2, "4x");

    m_ui.enumAdd(GUI_SUPERSAMPLE, 720, "720p");

    m_ui.enumAdd(GUI_SUPERSAMPLE, 1080, "1080p");

    m_ui.enumAdd(GUI_SUPERSAMPLE, 1440, "1440p");

    m_ui.enumAdd(GUI_SUPERSAMPLE, 2160, "2160p");

    m_ui.enumAdd(GUI_SUPERSAMPLE, 1024, "1024 sq");

    m_ui.enumAdd(GUI_SUPERSAMPLE, 2048, "2048 sq");

    m_ui.enumAdd(GUI_SUPERSAMPLE, 4096, "4096 sq");

    m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_MATERIAL, "material");

    m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_GREY, "grey");

    m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_VIS_BUFFER, "visibility buffer");

    m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_CLUSTER, "clusters");

    m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_GROUP, "cluster groups");

    m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_LOD, "lod levels");

    m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_TRIANGLE, "triangles");

    m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_SEMANTIC_LOD, "semantic lod policy");

    m_ui.enumAdd(GUI_SEMANTIC_LOD_MODE, int(SEMANTIC_LOD_POINTNEXT), "PointNeXt");
    m_ui.enumAdd(GUI_SEMANTIC_LOD_MODE, int(SEMANTIC_LOD_POINTCLIP), "PointCLIP");
    m_ui.enumAdd(GUI_SEMANTIC_LOD_MODE, int(SEMANTIC_LOD_POINTNEXT_STRUCTURE), "PointNeXt + structure");
    m_ui.enumAdd(GUI_SEMANTIC_LOD_MODE, int(SEMANTIC_LOD_POINTCLIP_STRUCTURE), "PointCLIP + structure");
    m_ui.enumAdd(GUI_SEMANTIC_LOD_MODE, int(SEMANTIC_LOD_POINTNEXT_POINTCLIP), "PointNeXt + PointCLIP");
    m_ui.enumAdd(GUI_SEMANTIC_LOD_MODE, int(SEMANTIC_LOD_POINTNEXT_POINTCLIP_STRUCTURE), "PointNeXt + PointCLIP + structure");
    m_ui.enumAdd(GUI_SEMANTIC_LOD_MODE, int(SEMANTIC_LOD_POINTNEXT_POINTCLIP_STRUCTURE_METADATA), "PointNeXt + open semantics + structure + CAD context");

  }


  m_profilerGpuTimer.init(m_profilerTimeline, app->getDevice(), app->getPhysicalDevice(), app->getQueue(0).familyIndex, true);
  m_resources.init(app->getDevice(), app->getPhysicalDevice(), app->getInstance(), app->getQueue(0), app->getQueue(1));

  {
    NVVK_CHECK(m_resources.m_samplerPool.acquireSampler(m_imguiSampler));

    NVVK_DBG_NAME(m_imguiSampler);
  }

  m_resources.initFramebuffer({128, 128}, m_tweak.supersample);

  updateImguiImage();


  setFromClusterConfig(m_sceneConfig, m_tweak.clusterConfig);

  if(!m_resources.m_supportsMeshShaderNV)
  {
    m_rendererConfig.useEXTmeshShader = true;
  }


  if(m_sceneFilePathDropNew.empty())
  {

    const std::filesystem::path              exeDirectoryPath   = nvutils::getExecutablePath().parent_path();
    const std::vector<std::filesystem::path> defaultSearchPaths = {

        std::filesystem::absolute(exeDirectoryPath / TARGET_EXE_TO_DOWNLOAD_DIRECTORY),

        std::filesystem::absolute(exeDirectoryPath / "resources"),
    };


    m_sceneFilePathDefault = m_sceneFilePathDropNew = nvutils::findFile("bunny_v2/bunny.gltf", defaultSearchPaths);


    m_sceneGridConfig.uniqueGeometriesForCopies = false;

    if(m_sceneGridConfig.numCopies == 1)
    {
      if(m_resources.getDeviceLocalHeapSize() >= 8ull * 1024 * 1024 * 1024)
      {
        m_sceneGridConfig.numCopies = 1;
      }
      else
      {
        m_sceneGridConfig.numCopies =1;
      }
    }
  }

  m_cameraStringCommandLine = m_cameraString;

  std::filesystem::path newFileDrop = m_sceneFilePathDropNew;

  onFileDrop(newFileDrop);

  m_tweakLast          = m_tweak;
  m_sceneConfigLast    = m_sceneConfig;
  m_sceneConfigEdit    = m_sceneConfig;
  m_rendererConfigLast = m_rendererConfig;
}


void LodClusters::onDetach()
{
  NVVK_CHECK(vkDeviceWaitIdle(m_app->getDevice()));


  deinitRenderer();

  deinitScene();


  m_resources.m_samplerPool.releaseSampler(m_imguiSampler);

  ImGui_ImplVulkan_RemoveTexture(m_imguiTexture);


  m_resources.deinit();


  m_profilerGpuTimer.deinit();
}


void LodClusters::parameterSequenceCallback(const nvutils::ParameterSequencer::State& state)
{
  std::string message;
  message += fmt::format("MemoryReport {} \"{}\" = {{ \n", state.index, state.description);
  if(m_renderer)
  {

    Renderer::ResourceUsageInfo resourceActual   = m_renderer->getResourceUsage(false);

    Renderer::ResourceUsageInfo resourceReserved = m_renderer->getResourceUsage(true);
    message += fmt::format("Memory; Actual; Reserved;\n");
    message += fmt::format("Geometry; {}; {};\n", resourceActual.geometryMemBytes, resourceReserved.geometryMemBytes);
    if(m_renderScene->useStreaming)
    {
      StreamingStats stats;

      m_renderScene->sceneStreaming.getStats(stats);
      message += fmt::format("Resident; Actual; Reserved;\n");
      message += fmt::format("Groups; {}; {};\n", stats.residentGroups, stats.maxGroups);
      message += fmt::format("Clusters; {}; {};\n", stats.residentClusters, stats.maxClusters);
    }

    shaderio::Readback readback;

    m_resources.getReadbackData(readback);
    message += fmt::format("Traversal; Actual; Reserved;\n");
    message += fmt::format("Traversal Tasks; {}; {};\n", readback.numTraversalTasks, m_renderer->getMaxTraversalTasks());
    message += fmt::format("Traversal Clusters; {}; {};\n", readback.numRenderClusters, m_renderer->getMaxRenderClusters());

  }
  message += fmt::format("}}\n");

  nvutils::Logger::getInstance().log(nvutils::Logger::eSTATS, "%s", message.c_str());

  if(m_sequenceScreenshotMode != SCREENSHOT_OFF)
  {
    ScreenshotMode screenshotMode = m_sequenceScreenshotMode;
    if(m_app->isHeadless())
    {
      screenshotMode = SCREENSHOT_VIEWPORT;
    }

    std::string filename = fmt::format("screenshot_{}_{}.jpg", state.index, state.description);
    if(screenshotMode == SCREENSHOT_WINDOW)
    {
      m_app->saveScreenShot(std::filesystem::path(filename), 100);
    }
    else if(screenshotMode == SCREENSHOT_VIEWPORT)
    {
      m_app->saveImageToFile(m_resources.m_frameBuffer.useResolved ? m_resources.m_frameBuffer.imgColorResolved.image :
                                                                     m_resources.m_frameBuffer.imgColor.image,
                             m_resources.m_frameBuffer.windowSize, filename, 100, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
  }
}

}
