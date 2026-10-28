//==============================================================================
// src/app/lodclusters_runtime.cpp
// Runs per-frame state reconciliation and command recording.
// Camera freezing, two-pass culling matrices, readback interpretation, and renderer dispatch are consolidated here before GPU submission.
//==============================================================================
#include <volk.h>
#include <nvgui/camera.hpp>
#include "lodclusters.hpp"


namespace lodclusters {


void LodClusters::onPreRender()
{

  m_profilerTimeline->frameAdvance();
}


void LodClusters::handleChanges()
{
  if(m_sceneLoading)
    return;

  m_sceneGridConfig.numCopies = std::clamp(m_sceneGridConfig.numCopies, SceneGridConfig::minCopies, SceneGridConfig::maxCopies);

  if(m_sceneFilePathDropLast != m_sceneFilePathDropNew)
  {
    std::filesystem::path newFilePath = m_sceneFilePathDropNew;

    onFileDrop(newFilePath);
  }

  if(!m_resources.m_supportsMeshShaderNV)
  {
    m_rendererConfig.useEXTmeshShader = true;
  }


  if((m_frameConfig.visualize == VISUALIZE_VIS_BUFFER || m_frameConfig.visualize == VISUALIZE_DEPTH_ONLY)
     && m_rendererConfig.useShading)
  {
    m_rendererConfig.useShading = false;
  }
  if(!(m_frameConfig.visualize == VISUALIZE_VIS_BUFFER || m_frameConfig.visualize == VISUALIZE_DEPTH_ONLY)
     && !m_rendererConfig.useShading)
  {
    m_rendererConfig.useShading = true;
  }
  m_rendererConfig.useDepthOnly = m_frameConfig.visualize == VISUALIZE_DEPTH_ONLY;

  bool frameBufferChanged = false;
  if(tweakChanged(m_tweak.supersample))
  {

    m_resources.initFramebuffer(m_windowSize, m_tweak.supersample);

    updateImguiImage();

    frameBufferChanged = true;
  }

  bool shaderChanged = false;
  if(m_reloadShaders)
  {
    shaderChanged   = true;
    m_reloadShaders = false;
  }

  bool sceneChanged = false;
  bool sceneReloadRequested = m_reloadScene;
  m_reloadScene = false;
  if(sceneReloadRequested || memcmp(&m_sceneConfig, &m_sceneConfigLast, sizeof(m_sceneConfig)))
  {
    sceneChanged = true;


    deinitRenderer();

    initScene(m_sceneFilePath, m_scene->m_cacheSuffix, true);
  }

  if(!m_cameraString.empty() && m_cameraString != m_cameraStringLast)
  {

    applyCameraString();
  }

  bool sceneGridChanged = false;
  if(m_scene)
  {
    if(!m_renderScene)
    {
      sceneGridChanged = true;
    }

    if(!sceneChanged && memcmp(&m_sceneGridConfig, &m_sceneGridConfigLast, sizeof(m_sceneGridConfig)))
    {
      sceneGridChanged = true;


      deinitRenderer();

      m_scene->updateSceneGrid(m_sceneGridConfig);

      updatedSceneGrid();
    }

    bool renderSceneChanged = false;

    bool streamingChanged   = tweakChanged(m_tweak.useStreaming)
                            || (memcmp(&m_streamingConfig, &m_streamingConfigLast, sizeof(m_streamingConfig)));
    if(sceneGridChanged || streamingChanged)
    {
      if(!sceneChanged || !sceneGridChanged)
      {

        deinitRenderer();
      }

      renderSceneChanged = true;

      deinitRenderScene();

      initRenderScene();

      if(streamingChanged)
      {
        m_streamGeometryHistogramMax = 0;
      }
    }

    if(sceneChanged || shaderChanged || renderSceneChanged || tweakChanged(m_tweak.renderer) || tweakChanged(m_tweak.supersample)

       || rendererCfgChanged(m_rendererConfig.flipWinding) || rendererCfgChanged(m_rendererConfig.useDebugVisualization)

       || rendererCfgChanged(m_rendererConfig.useCulling) || rendererCfgChanged(m_rendererConfig.forceTwoSided)

       || rendererCfgChanged(m_rendererConfig.useSorting) || rendererCfgChanged(m_rendererConfig.numRenderClusterBits)

       || rendererCfgChanged(m_rendererConfig.numTraversalTaskBits) || rendererCfgChanged(m_rendererConfig.useShading)

       || rendererCfgChanged(m_rendererConfig.useRenderStats)

       || rendererCfgChanged(m_rendererConfig.useSeparateGroups)

       || rendererCfgChanged(m_rendererConfig.useEXTmeshShader)

       || rendererCfgChanged(m_rendererConfig.useTwoPassCulling)
       || rendererCfgChanged(m_rendererConfig.useAdaptiveTwoPassCulling)
       || rendererCfgChanged(m_rendererConfig.twoPassMinClusters)
       || rendererCfgChanged(m_rendererConfig.twoPassMatrixDelta) || rendererCfgChanged(m_rendererConfig.useDepthOnly)
       || rendererCfgChanged(m_rendererConfig.adaptiveTwoPassMode) || rendererCfgChanged(m_rendererConfig.twoPassMotionPixels)
       || rendererCfgChanged(m_rendererConfig.twoPassFastPixels) || rendererCfgChanged(m_rendererConfig.twoPassMaxInterval)
       || rendererCfgChanged(m_rendererConfig.semanticRuntimeStrength) || rendererCfgChanged(m_rendererConfig.semanticRuntimePivot)
       || rendererCfgChanged(m_rendererConfig.useParallelGroups) || rendererCfgChanged(m_rendererConfig.boundAssemblyLod)
       || rendererCfgChanged(m_rendererConfig.useAssemblyCompact) || rendererCfgChanged(m_rendererConfig.useAssemblyEarlyExit) || rendererCfgChanged(m_rendererConfig.useAssemblyPrefilter) || rendererCfgChanged(m_rendererConfig.useLowDetailFastPath)
       || rendererCfgChanged(m_rendererConfig.useHizOcclusion)
       || rendererCfgChanged(m_rendererConfig.groupClusterLanes) || rendererCfgChanged(m_rendererConfig.twoPassCostProbeInterval)
       || rendererCfgChanged(m_rendererConfig.twoPassCostHysteresis) || rendererCfgChanged(m_rendererConfig.twoPassCostSmoothing)
       || rendererCfgChanged(m_rendererConfig.twoPassCostWarmupFrames)
       || rendererCfgChanged(m_rendererConfig.useForcedInvisibleCulling))
    {


      initRenderer(m_tweak.renderer);
    }
    else if(m_renderer && frameBufferChanged)
    {

      m_renderer->updatedFrameBuffer(m_resources, *m_renderScene);
      m_rendererFboChangeID = m_resources.m_fboChangeID;
    }
  }


  bool hadChange = sceneReloadRequested || shaderChanged || memcmp(&m_tweakLast, &m_tweak, sizeof(m_tweak))
                   || memcmp(&m_rendererConfigLast, &m_rendererConfig, sizeof(m_rendererConfig))
                   || memcmp(&m_sceneConfigLast, &m_sceneConfig, sizeof(m_sceneConfig))
                   || memcmp(&m_streamingConfigLast, &m_streamingConfig, sizeof(m_streamingConfig))
                   || memcmp(&m_sceneGridConfigLast, &m_sceneGridConfig, sizeof(m_sceneGridConfig));
  m_tweakLast           = m_tweak;
  m_rendererConfigLast  = m_rendererConfig;
  m_streamingConfigLast = m_streamingConfig;
  m_sceneConfigLast     = m_sceneConfig;
  m_sceneGridConfigLast = m_sceneGridConfig;

  if(hadChange)
  {
    m_equalFrames = 0;
    if(m_tweak.autoResetTimers)
    {

      m_info.profilerManager->resetFrameSections(8);
    }
  }
}


void LodClusters::applyCameraString()
{

  nvutils::CameraManipulator::Camera cam = m_info.cameraManipulator->getCamera();
  if(cam.setFromString(m_cameraString))
  {

    m_info.cameraManipulator->setCamera(cam);
    nvgui::SetHomeCamera(m_info.cameraManipulator->getCamera());
  }
  m_cameraStringLast = m_cameraString;
}


void LodClusters::onRender(VkCommandBuffer cmd)
{

  double time = m_clock.getSeconds();
  static double lastTime = 0.0;
  float deltaTime = static_cast<float>(time - lastTime);
  lastTime = time;


  m_resources.beginFrame(m_app->getFrameCycleIndex());
  if(!m_benchOutput.empty() && m_renderer)
    benchmarkPrepare();


  m_frameConfig.windowSize = m_windowSize;


  if(m_renderer)
  {

    if(m_rendererFboChangeID != m_resources.m_fboChangeID)
    {

      m_renderer->updatedFrameBuffer(m_resources, *m_renderScene);
      m_rendererFboChangeID = m_resources.m_fboChangeID;
    }


    shaderio::FrameConstants& frameConstants = m_frameConfig.frameConstants;


    frameConstants.viewProjMatrixPrev = frameConstants.viewProjMatrix;

    if(m_frames)
    {
      m_frameConfig.frameConstantsLast = m_frameConfig.frameConstants;
    }

    int supersample = m_tweak.supersample;


    uint32_t renderWidth  = m_resources.m_frameBuffer.renderSize.width;
    uint32_t renderHeight = m_resources.m_frameBuffer.renderSize.height;

    uint32_t targetWidth  = m_resources.m_frameBuffer.targetSize.width;
    uint32_t targetHeight = m_resources.m_frameBuffer.targetSize.height;

    frameConstants.facetShading = m_tweak.facetShading ? 1 : 0;
    frameConstants.visualize    = m_frameConfig.visualize;
    frameConstants.frame        = m_frames;


    static float accumulatedTime = 0.0f;
    accumulatedTime += deltaTime;
    frameConstants.time = accumulatedTime;
    frameConstants.deltaTime = deltaTime;

    {
      frameConstants.visFilterClusterID  = ~0;
      frameConstants.visFilterInstanceID = ~0;
    }

    frameConstants.bgColor   = m_resources.m_bgColor;

    frameConstants.viewport  = glm::ivec2(renderWidth, renderHeight);

    frameConstants.viewportf = glm::vec2(renderWidth, renderHeight);

    frameConstants.nearPlane = m_info.cameraManipulator->getClipPlanes().x;
    frameConstants.farPlane  = m_info.cameraManipulator->getClipPlanes().y;

    frameConstants.wUpDir    = m_info.cameraManipulator->getUp();
    frameConstants.fov = glm::radians(m_info.cameraManipulator->getFov());


    glm::mat4 projection = glm::perspectiveRH_ZO(frameConstants.fov, float(targetWidth) / float(targetHeight), frameConstants.farPlane, frameConstants.nearPlane);
    projection[1][1] *= -1;

    glm::mat4 view  = m_info.cameraManipulator->getViewMatrix();

    glm::mat4 viewI = glm::inverse(view);

    frameConstants.viewProjMatrix  = projection * view;

    frameConstants.viewProjMatrixI = glm::inverse(frameConstants.viewProjMatrix);
    frameConstants.viewMatrix      = view;
    frameConstants.viewMatrixI     = viewI;
    frameConstants.projMatrix      = projection;

    frameConstants.projMatrixI     = glm::inverse(projection);

    glm::mat4 viewNoTrans         = view;
    viewNoTrans[3]                = {0.0f, 0.0f, 0.0f, 1.0f};

    frameConstants.skyProjMatrixI = glm::inverse(projection * viewNoTrans);


    glm::vec4 hPos   = projection * glm::vec4(1.0f, 1.0f, -frameConstants.farPlane, 1.0f);

    glm::vec2 hCoord = glm::vec2(hPos.x / hPos.w, hPos.y / hPos.w);

    glm::vec2 dim    = glm::abs(hCoord);


    frameConstants.viewPixelSize = dim * (glm::vec2(float(renderWidth), float(renderHeight)) * 0.5f) * frameConstants.farPlane;


    frameConstants.viewClipSize = dim * frameConstants.farPlane;

    frameConstants.viewPos = frameConstants.viewMatrixI[3];
    frameConstants.viewDir = -viewI[2];

    frameConstants.viewPlane   = frameConstants.viewDir;
    frameConstants.viewPlane.w = -glm::dot(glm::vec3(frameConstants.viewPos), glm::vec3(frameConstants.viewDir));

    frameConstants.wLightPos = frameConstants.viewMatrixI[3];

    {


      m_resources.m_hizUpdate[0].farInfo.getShaderFactors((float*)&frameConstants.hizSizeFactors);

      frameConstants.hizSizeMax = m_resources.m_hizUpdate[0].farInfo.getSizeMax();


    }


    if(!m_frames)
    {

      m_frameConfig.frameConstantsLast = m_frameConfig.frameConstants;
    }

    if(!m_frameConfig.freezeLoD)
    {
      m_frameConfig.traversalViewMatrix = m_frameConfig.frameConstants.viewMatrix;
    }
    if(!m_frameConfig.freezeCulling)
    {
      m_frameConfig.cullViewProjMatrix     = m_frameConfig.frameConstants.viewProjMatrix;
      m_frameConfig.cullViewProjMatrixLast = m_frameConfig.frameConstantsLast.viewProjMatrix;
    }

    if(m_frames)
    {
      shaderio::FrameConstants frameCurrent = m_frameConfig.frameConstants;

      if(memcmp(&frameCurrent, &m_frameConfig.frameConstantsLast, sizeof(shaderio::FrameConstants)))
        m_equalFrames = 0;
      else
        m_equalFrames++;
    }


    if(m_resources.benchmarkQueries != VK_NULL_HANDLE)
    {
      vkCmdResetQueryPool(cmd, m_resources.benchmarkQueries, m_resources.getFrameCycleIndex()*24, 24);
      m_resources.benchmarkTimestamp(cmd, 0, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);
    }
    m_renderer->render(cmd, m_resources, *m_renderScene, m_frameConfig, m_profilerGpuTimer);
  }
  else
  {

    m_resources.emptyFrame(cmd, m_frameConfig, m_profilerGpuTimer);
  }

  {

    m_resources.postProcessFrame(cmd, m_frameConfig, m_profilerGpuTimer);
  }


  m_resources.endFrame();
  if(m_resources.benchmarkQueries != VK_NULL_HANDLE && m_renderer)
  {
    m_resources.benchmarkTimestamp(cmd, 1);
    benchmarkCapture(cmd);
  }


  VkSemaphoreSubmitInfo semSubmit = m_resources.m_queueStates.primary.advanceSignalSubmit(VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT);

  m_app->addSignalSemaphore(semSubmit);

  while(!m_resources.m_queueStates.primary.m_pendingWaits.empty())
  {
    m_app->addWaitSemaphore(m_resources.m_queueStates.primary.m_pendingWaits.back());

    m_resources.m_queueStates.primary.m_pendingWaits.pop_back();
  }

  m_lastTime = time;
  m_frames++;
}

}

