//==============================================================================
// src/app/main.cpp
// Application entry point for the LOD clusters sample.
// Creates the Vulkan context, installs the app element, and forwards capability flags discovered at startup.
//==============================================================================



#ifndef NDEBUG


#define VMA_LEAK_LOG_FORMAT(format, ...)                                                                               \
  do                                                                                                                   \
  {                                                                                                                    \
    fprintf(stderr, (format), __VA_ARGS__);                                                                            \
    fprintf(stderr, "\n");                                                                                             \
  } while(false)
#endif


#define VMA_IMPLEMENTATION

#if __INTELLISENSE__
#undef VK_NO_PROTOTYPES
#endif


#include <volk.h>
#include <imgui/imgui.h>
#include <nvvk/validation_settings.hpp>
#include <nvapp/elem_logger.hpp>
#include <nvapp/elem_profiler.hpp>
#include <nvapp/elem_camera.hpp>
#include <nvapp/elem_default_menu.hpp>
#include <nvapp/elem_default_title.hpp>
#include <nvapp/elem_sequencer.hpp>
#include <nvutils/parameter_parser.hpp>

#include "lodclusters.hpp"

using namespace lodclusters;

// Opt-in for offline comparisons: exclude asynchronous loading/blank frames from
// the sequencer, including its warmup and screenshots. Normal scripts may load
// their first scene themselves and therefore retain the original behavior.
class SceneReadySequencer : public nvapp::ElementSequencer
{
public:
  SceneReadySequencer(const nvutils::ParameterSequencer::InitInfo& info, std::weak_ptr<LodClusters> scene, bool wait)
      : nvapp::ElementSequencer(info), m_scene(scene), m_wait(wait) {}
  void onPreRender() override
  {
    auto scene = m_scene.lock();
    if(!m_wait || (scene && scene->isRenderReadyForSequence()))
      nvapp::ElementSequencer::onPreRender();
  }
private:
  std::weak_ptr<LodClusters> m_scene;
  bool m_wait;
};


int main(int argc, char** argv)
{
  nvapp::ApplicationCreateInfo appInfo;
  appInfo.name    = TARGET_NAME;
  appInfo.useMenu = true;
  appInfo.vSync = false;
  VkPhysicalDeviceShaderSMBuiltinsFeaturesNV smNV = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SM_BUILTINS_FEATURES_NV};
  VkPhysicalDeviceMeshShaderFeaturesNV       meshNV  = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_NV};
  VkPhysicalDeviceMeshShaderFeaturesEXT      meshEXT = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
  VkPhysicalDeviceShaderClockFeaturesKHR clockKHR = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR};
  VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomicFloatFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT};
  VkPhysicalDeviceFragmentShadingRateFeaturesKHR shadingRateFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_FEATURES_KHR};
  VkPhysicalDeviceFragmentShaderBarycentricFeaturesKHR barycentricFeatures{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR};
  VkPhysicalDeviceShaderImageAtomicInt64FeaturesEXT shaderImageAtomic64Features{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_IMAGE_ATOMIC_INT64_FEATURES_EXT};

  nvvk::ContextInitInfo vkSetup{
      .instanceExtensions = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME},
      .deviceExtensions   = {{VK_KHR_SWAPCHAIN_EXTENSION_NAME}},
      .queues             = {VK_QUEUE_GRAPHICS_BIT, VK_QUEUE_TRANSFER_BIT},
  };

  vkSetup.deviceExtensions.push_back({VK_EXT_MESH_SHADER_EXTENSION_NAME, &meshEXT});
  vkSetup.deviceExtensions.push_back({VK_KHR_SHADER_CLOCK_EXTENSION_NAME, &clockKHR});
  vkSetup.deviceExtensions.push_back({VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME, &atomicFloatFeatures});


  vkSetup.deviceExtensions.push_back({VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME, &shadingRateFeatures});

  vkSetup.deviceExtensions.push_back({VK_EXT_SHADER_IMAGE_ATOMIC_INT64_EXTENSION_NAME, &shaderImageAtomic64Features});
#if 1


  vkSetup.deviceExtensions.push_back({VK_NV_SHADER_SM_BUILTINS_EXTENSION_NAME, &smNV, false});
  vkSetup.deviceExtensions.push_back({VK_NV_MESH_SHADER_EXTENSION_NAME, &meshNV, false});

  vkSetup.deviceExtensions.push_back({VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME, &barycentricFeatures, false});
#endif


  nvutils::ProfilerManager                    profilerManager;
  std::shared_ptr<nvutils::CameraManipulator> cameraManipulator = std::make_shared<nvutils::CameraManipulator>();

  nvutils::ParameterRegistry            parameterRegistry;
  nvutils::ParameterParser              parameterParser;
  nvutils::ParameterSequencer::InitInfo sequencerInfo{
                                                      .parameterParser   = &parameterParser,
                                                      .parameterRegistry = &parameterRegistry,

                                                      .profilerManager = &profilerManager};

  nvvk::ValidationSettings::LayerPresets validationPreset = nvvk::ValidationSettings::LayerPresets::eStandard;

  parameterRegistry.add({"validation"}, &vkSetup.enableValidationLayers);
  parameterRegistry.add({"validationpreset"}, (int*)&validationPreset);
  parameterRegistry.add({"vsync"}, &appInfo.vSync);
  parameterRegistry.add({"device", "force a vulkan device via index into the device list"}, &vkSetup.forceGPU);
  parameterRegistry.add({"headless"}, &appInfo.headless, true);
  parameterRegistry.add({"headlessframes"}, &appInfo.headlessFrameCount);
  parameterRegistry.add({"headlesswidth"}, &appInfo.windowSize.x);
  parameterRegistry.add({"headlessheight"}, &appInfo.windowSize.y);
  bool sequenceWaitForScene = false;
  parameterRegistry.add({"sequencewaitforscene", "start sequence measurements only after the scene is rendered"}, &sequenceWaitForScene);

  LodClusters::Info sampleInfo;
  sampleInfo.cameraManipulator               = cameraManipulator;
  sampleInfo.profilerManager                 = &profilerManager;
  sampleInfo.parameterRegistry               = &parameterRegistry;
  sampleInfo.parameterParser                 = &parameterParser;
  std::shared_ptr<LodClusters> sampleElement = std::make_shared<LodClusters>(sampleInfo);


  sequencerInfo.registerScriptParameters(parameterRegistry, parameterParser);


  sequencerInfo.postCallbacks.emplace_back(
      [&](const nvutils::ParameterSequencer::State& state) { sampleElement->parameterSequenceCallback(state); });


  parameterParser.add(parameterRegistry);

  parameterParser.setVerbose(true);

  parameterParser.parse(argc, argv);


  auto elemSequencer = std::make_shared<SceneReadySequencer>(sequencerInfo, sampleElement, sequenceWaitForScene);

  if(sampleElement->isCacheLoadOnly())
  {
    sampleElement->doCacheLoadOnly();
    return 0;
  }


  if(sampleElement->isProcessingOnly())
  {

    sampleElement->doProcessingOnly();
    return 0;
  }

  nvvk::ValidationSettings validationSettings;
  if(vkSetup.enableValidationLayers)
  {

    validationSettings.setPreset(validationPreset);
    validationSettings.duplicate_message_limit = 3;
    validationSettings.message_id_filter = {"VUID-RuntimeSpirv-storageInputOutput16-06334", "VUID-VkShaderModuleCreateInfo-pCode-08740"};


    vkSetup.instanceCreateInfoExt = validationSettings.buildPNextChain();
  }


  nvvk::addSurfaceExtensions(vkSetup.instanceExtensions);
  nvvk::Context vkContext;


  NVVK_CHECK(volkInitialize());

  {


    nvutils::ScopedTimer st("Creating Vulkan Context");
    VkResult result{};
    vkContext.contextInfo = vkSetup;

    result = vkContext.createInstance();

    result = vkContext.selectPhysicalDevice();

    result = vkContext.createDevice();

    NVVK_CHECK(result);
    nvvk::DebugUtil::getInstance().init(vkContext.getDevice());
    if(vkContext.contextInfo.verbose)
    {
      NVVK_CHECK(nvvk::Context::printVulkanVersion());
      NVVK_CHECK(nvvk::Context::printInstanceLayers());
      NVVK_CHECK(nvvk::Context::printInstanceExtensions(vkContext.contextInfo.instanceExtensions));
      NVVK_CHECK(nvvk::Context::printDeviceExtensions(vkContext.getPhysicalDevice(), vkContext.contextInfo.deviceExtensions));
    }
    {
      NVVK_CHECK(nvvk::Context::printGpus(vkContext.getInstance(), vkContext.getPhysicalDevice()));

      LOGI("_________________________________________________\n");
    }
  }

  sampleElement->setSupportsBarycentrics(vkContext.hasExtensionEnabled(VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME));
  sampleElement->setSupportsMeshShaderNV(vkContext.hasExtensionEnabled(VK_NV_MESH_SHADER_EXTENSION_NAME));
  sampleElement->setSupportsSmBuiltinsNV(vkContext.hasExtensionEnabled(VK_NV_SHADER_SM_BUILTINS_EXTENSION_NAME));

  appInfo.instance       = vkContext.getInstance();

  appInfo.device         = vkContext.getDevice();

  appInfo.physicalDevice = vkContext.getPhysicalDevice();

  appInfo.queues         = vkContext.getQueueInfos();


  bool hasDebugUI = sampleElement->getShowDebugUI();


  appInfo.dockSetup = [&hasDebugUI](ImGuiID viewportID) {
    if(hasDebugUI)
    {


      ImGuiID debugID = ImGui::DockBuilderSplitNode(viewportID, ImGuiDir_Left, 0.15F, nullptr, &viewportID);

      ImGui::DockBuilderDockWindow("Debug", debugID);
    }


    ImGuiID settingID = ImGui::DockBuilderSplitNode(viewportID, ImGuiDir_Right, 0.25F, nullptr, &viewportID);

    ImGui::DockBuilderDockWindow("Settings", settingID);

    ImGui::DockBuilderDockWindow("Misc Settings", settingID);


    ImGuiID loggerID = ImGui::DockBuilderSplitNode(viewportID, ImGuiDir_Down, 0.35F, nullptr, &viewportID);

    ImGui::DockBuilderDockWindow("Log", loggerID);

    ImGuiID profilerID = ImGui::DockBuilderSplitNode(loggerID, ImGuiDir_Right, 0.75F, nullptr, &loggerID);

    ImGui::DockBuilderDockWindow("Profiler", profilerID);

    ImGuiID streamingID = ImGui::DockBuilderSplitNode(profilerID, ImGuiDir_Right, 0.66F, nullptr, &profilerID);

    ImGui::DockBuilderDockWindow("Streaming memory", streamingID);

    ImGuiID statisticsID = ImGui::DockBuilderSplitNode(streamingID, ImGuiDir_Right, 0.5F, nullptr, &streamingID);

    ImGui::DockBuilderDockWindow("Statistics", statisticsID);
  };


  nvapp::Application app;

  app.init(appInfo);

  auto                  logger      = std::make_shared<nvapp::ElementLogger>();

  nvapp::ElementLogger* loggerDeref = logger.get();
  nvutils::Logger::getInstance().setLogCallback([&](nvutils::Logger::LogLevel logLevel, const std::string& text) {
    loggerDeref->addLog(logLevel, "%s", text.c_str());
  });

  auto profilerUiSettings          = std::make_shared<nvapp::ElementProfiler::ViewSettings>();
  profilerUiSettings->table.levels = 1u;

  app.addElement(elemSequencer);
  app.addElement(std::make_shared<nvapp::ElementDefaultWindowTitle>());

  app.addElement(sampleElement);

  app.addElement(logger);
  app.addElement(std::make_shared<nvapp::ElementCamera>(cameraManipulator));
  app.addElement(std::make_shared<nvapp::ElementProfiler>(&profilerManager, profilerUiSettings));

  app.run();

  nvutils::Logger::getInstance().setLogCallback(nullptr);


  app.deinit();

  vkContext.deinit();

  return 0;
}
