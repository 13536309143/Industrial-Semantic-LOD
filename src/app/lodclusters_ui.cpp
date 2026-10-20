//==============================================================================
// src/app/lodclusters_ui.cpp
// Builds the ImGui controls and viewport overlays for the sample.
// UI edits are written to shadow config structs; the runtime step decides which changes require scene, streaming, or renderer rebuilds.
//==============================================================================
#include <cinttypes>
#include <filesystem>
#include <chrono>
#include <thread>
#include <array>
#include <unordered_set>
#include <imgui/imgui.h>
#include <imgui/imgui_internal.h>
#include <implot/implot.h>
#include <nvgui/fonts.hpp>
#include <nvgui/camera.hpp>
#include <nvgui/sky.hpp>
#include <nvgui/property_editor.hpp>
#include <nvgui/window.hpp>
#include <nvgui/file_dialog.hpp>
#include "lodclusters.hpp"


namespace lodclusters {


#define MEMORY_WITH_BINARY_PREFIXES 1


std::string formatMemorySize(size_t sizeInBytes)
{
#if MEMORY_WITH_BINARY_PREFIXES
  static const std::string units[]     = {"B", "KiB", "MiB", "GiB"};
  static const size_t      unitSizes[] = {1, 1024, 1024 * 1024, 1024 * 1024 * 1024};
#else
  static const std::string units[]     = {"B", "KB", "MB", "GB"};
  static const size_t      unitSizes[] = {1, 1000, 1000 * 1000, 1000 * 1000 * 1000};
#endif

  uint32_t currentUnit = 0;
  for(uint32_t i = 1; i < 4; i++)
  {
    if(sizeInBytes < unitSizes[i])
    {
      break;
    }
    currentUnit++;
  }


  float size = float(sizeInBytes) / float(unitSizes[currentUnit]);

  return fmt::format("{:.3} {}", size, units[currentUnit]);
}


std::string formatMetric(size_t size)
{


    return fmt::format("{}", size);
}

const char* semanticLodModeName(uint32_t mode)
{
  switch(mode)
  {
    case SEMANTIC_LOD_POINTNEXT:
      return "PointNeXt";
    case SEMANTIC_LOD_POINTCLIP:
      return "PointCLIP";
    case SEMANTIC_LOD_POINTNEXT_STRUCTURE:
      return "PointNeXt + structure";
    case SEMANTIC_LOD_POINTCLIP_STRUCTURE:
      return "PointCLIP + structure";
    case SEMANTIC_LOD_POINTNEXT_POINTCLIP:
      return "PointNeXt + PointCLIP";
    case SEMANTIC_LOD_POINTNEXT_POINTCLIP_STRUCTURE:
      return "PointNeXt + PointCLIP + structure";
    case SEMANTIC_LOD_POINTNEXT_POINTCLIP_STRUCTURE_METADATA:
      return "PointNeXt + open semantics + structure + CAD context";
    default:
      return "PointNeXt + PointCLIP + structure";
  }
}

template <typename T, typename Tcont>

void uiPlot(const std::string& plotName, const std::string& tooltipFormat, const Tcont& data, const T& maxValue, int offset = 0, size_t sizeOverride = 0)
{
  ImVec2 plotSize = ImVec2(ImGui::GetContentRegionAvail().x, ImGui::GetContentRegionAvail().y / 2);

  size_t size     = sizeOverride ? sizeOverride : data.size();


  plotSize.y = std::max(plotSize.y, ImGui::GetTextLineHeight() * 20);

  const ImPlotFlags     plotFlags = ImPlotFlags_NoBoxSelect | ImPlotFlags_NoMouseText | ImPlotFlags_Crosshairs;
  const ImPlotAxisFlags axesFlags = ImPlotAxisFlags_Lock | ImPlotAxisFlags_NoLabel;

  const ImColor         plotColor = ImColor(0.07f, 0.9f, 0.06f, 1.0f);

  if(ImPlot::BeginPlot(plotName.c_str(), plotSize, plotFlags))
  {

    ImPlot::SetupLegend(ImPlotLocation_NorthWest, ImPlotLegendFlags_NoButtons);

    ImPlot::SetupAxes(nullptr, "Count", axesFlags, axesFlags);
    ImPlot::SetupAxesLimits(0, double(size), 0, static_cast<double>(maxValue), ImPlotCond_Always);

    ImPlot::PushStyleVar(ImPlotStyleVar_FillAlpha, 0.25f);

    ImPlot::SetAxes(ImAxis_X1, ImAxis_Y1);

    ImPlot::SetNextFillStyle(plotColor);
    ImPlot::PlotShaded("", data.data(), (int)size, -INFINITY, 1.0, 0.0, 0, offset);

    ImPlot::PopStyleVar();
    if(ImPlot::IsPlotHovered())
    {

      ImPlotPoint mouse       = ImPlot::GetPlotMousePos();
      int         mouseOffset = (int(mouse.x)) % (int)size;

      ImGui::BeginTooltip();
      ImGui::Text(tooltipFormat.c_str(), mouseOffset, data[mouseOffset]);

      ImGui::EndTooltip();
    }


    ImPlot::EndPlot();
  }
}


static uint32_t getUsagePct(uint64_t requested, uint64_t reserved)
{
  bool     exceeds = requested > reserved;
  uint32_t pct     = uint32_t(double(requested) * 100.0 / double(std::max(reserved, uint64_t(1))));

  if(exceeds && pct < 101)
    pct = 101;
  return pct;
}


struct UsagePercentages
{
  uint32_t pctClusters  = 0;
  uint32_t pctTasks     = 0;
  uint32_t pctResident  = 0;
  uint32_t pctGeoMemory = 0;


  void setupPercentages(shaderio::Readback& readback, uint64_t maxRenderClusters, uint64_t maxTraversalTasks)
  {
    pctClusters = getUsagePct(readback.numRenderClusters, maxRenderClusters);

    pctTasks    = getUsagePct(readback.numTraversalTasks, maxTraversalTasks);
  }


  void setupPercentages(StreamingStats& stats, const StreamingConfig& streamingConfig)
  {
    pctResident = uint32_t(double(stats.residentGroups) * 100.0 / double(stats.maxGroups));
    pctGeoMemory = uint32_t(double(stats.usedDataBytes) * 100.0 / double(stats.maxDataBytes));
  }


  const char* getWarning()
  {
    if(pctClusters > 100)
      return "WARNING: Scene: Render clusters limit exceeded";
    if(pctTasks > 100)
      return "WARNING: Scene: Traversal task limit exceeded";
    if(pctResident == 100)
      return "WARNING: Streaming: No resident groups left";
    if(pctGeoMemory >= 99)
      return "WARNING: Streaming: Little geometry memory left";
    return nullptr;
  }
};


void LodClusters::viewportUI(ImVec2 corner)
{

  ImVec2     mouseAbsPos = ImGui::GetMousePos();
  glm::uvec2 mousePos    = {mouseAbsPos.x - corner.x, mouseAbsPos.y - corner.y};

  m_frameConfig.frameConstants.mousePosition = glm::uvec2(glm::vec2(mousePos) * m_resources.getFramebufferWindow2RenderScale());


  if(m_renderer)
  {
    shaderio::Readback readback;

    m_resources.getReadbackData(readback);

    UsagePercentages pct;

    if(m_renderScene->useStreaming)
    {
      StreamingStats streamingStats;

      m_renderScene->sceneStreaming.getStats(streamingStats);

      pct.setupPercentages(streamingStats, m_streamingConfig);
    }


    const char* warning = pct.getWarning();

    if(warning)
    {
      ImVec4 warn_color = {0.75f, 0.2f, 0.2f, 1};
      ImVec4 hi_color   = {0.85f, 0.3f, 0.3f, 1};
      ImVec4 lo_color   = {0, 0, 0, 1};


      ImGui::SetWindowFontScale(2.0);


      ImGui::SetCursorPos({7, 7});

      ImGui::TextColored(lo_color, "%s", warning);
      ImGui::SetCursorPos({9, 9});

      ImGui::TextColored(lo_color, "%s", warning);
      ImGui::SetCursorPos({9, 7});

      ImGui::TextColored(lo_color, "%s", warning);
      ImGui::SetCursorPos({7, 9});

      ImGui::TextColored(lo_color, "%s", warning);
      ImGui::SetCursorPos({8, 8});

      ImGui::TextColored(hi_color, "%s", warning);

      ImGui::SetWindowFontScale(1.0);
    }
  }
}


void LodClusters::loadingUI() {}


void LodClusters::onUIRender()
{

  ImGuiWindow* viewport = ImGui::FindWindowByName("Viewport");

  bool requestCameraRecenter = false;


  if(m_sceneLoading)
  {


    ImGui::OpenPopup("Busy Info");


    const ImVec2 win_size(300, 100);

    ImGui::SetNextWindowSize(win_size);

    const ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5F, 0.5F));


    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 15.0);
    if(ImGui::BeginPopupModal("Busy Info", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoDecoration))
    {


      ImGui::TextDisabled("Please wait ...");

      ImGui::NewLine();
      ImGui::ProgressBar(float(m_sceneProgress) / 100.0f, ImVec2(-1.0f, 0.0f), "Loading Scene");

      ImGui::EndPopup();
    }

    ImGui::PopStyleVar();
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }

  if(viewport)
  {
    if(nvgui::isWindowHovered(viewport))
    {
      if(ImGui::IsKeyDown(ImGuiKey_R))
      {
        m_reloadShaders = true;
      }
      if(ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left) || ImGui::IsKeyPressed(ImGuiKey_Space))
      {
        requestCameraRecenter = true;
      }
    }
  }

  shaderio::Readback readback;

  m_resources.getReadbackData(readback);


  bool pickingValid = isPickingValid(readback);


  ImVec4 text_color = ImGui::GetStyleColorVec4(ImGuiCol_Text);
  ImVec4 warn_color = text_color;
  warn_color.y *= 0.5f;
  warn_color.z *= 0.5f;


  const ImVec4 recommendedColor = ImVec4(0.0f, 0.4f, 0.8f, 1.0f);

  const ImVec4 changesColor     = ImVec4(0.8f, 0.4f, 0.0f, 1.0f);

  UsagePercentages pct = {};
  if(m_renderer)
  {
    pct.setupPercentages(readback, m_renderer->getMaxRenderClusters(), m_renderer->getMaxTraversalTasks());
  }

  StreamingStats stats = {};
  if(m_renderScene && m_renderScene->useStreaming)
  {

    m_renderScene->sceneStreaming.getStats(stats);
  }

  namespace PE = nvgui::PropertyEditor;

  if(ImGui::Begin("Settings"))
  {
    ImGui::PushItemWidth(170 * ImGui::GetWindowDpiScale());


    if(ImGui::CollapsingHeader("Rendering", nullptr, ImGuiTreeNodeFlags_DefaultOpen))
    {


      PE::begin("##Rendering", ImGuiTableFlags_Resizable);

      PE::entry("Super Resolution",[&]() { return m_ui.enumCombobox(GUI_SUPERSAMPLE, "sampling", &m_tweak.supersample); });

      PE::Text("Render Resolution:", "%d x %d", m_resources.m_frameBuffer.renderSize.width,m_resources.m_frameBuffer.renderSize.height);

      ImGui::PushStyleColor(ImGuiCol_Text, recommendedColor);
      PE::entry("Visualize", [&]() {

        ImGui::PopStyleColor();
        return m_ui.enumCombobox(GUI_VISUALIZE, "visualize", &m_frameConfig.visualize);
      });
      if(m_tweak.renderer == RENDERER_RASTER_CLUSTERS_LOD)

      PE::Checkbox("Instance BBoxes", &m_frameConfig.showInstanceBboxes);

      PE::Checkbox("Cluster BBoxes", &m_frameConfig.showClusterBboxes);

      PE::end();
    }
    if(ImGui::CollapsingHeader("Traversal", nullptr, ImGuiTreeNodeFlags_DefaultOpen))
    {

      PE::begin("##TraversalSpecifics", ImGuiTableFlags_Resizable);


      PE::InputFloat("LoD pixel error", &m_frameConfig.lodPixelError, 0.25f, 0.25f, "%.3f", ImGuiInputTextFlags_EnterReturnsTrue);

      m_frameConfig.lodPixelError = std::max(0.000f, m_frameConfig.lodPixelError);
      if(PE::treeNode("Other settings"))
      {
        PE::Checkbox("Separate Groups Kernel", &m_rendererConfig.useSeparateGroups,
                     "optimization that splits traversal into two separate kernels");

        PE::Checkbox("Instance Sorting", &m_rendererConfig.useSorting);
        PE::Checkbox("Enqueued Statistics", &m_rendererConfig.useRenderStats,
                     "Adds additional atomic counters for statistics, impacts performance");
        PE::Checkbox("Culling (Occlusion & Frustum)", &m_rendererConfig.useCulling);

        ImGui::BeginDisabled(!m_rendererConfig.useCulling);


        PE::Checkbox("Freeze Culling", &m_frameConfig.freezeCulling);

        PE::Checkbox("Freeze LoD", &m_frameConfig.freezeLoD);
        if(m_tweak.renderer == RENDERER_RASTER_CLUSTERS_LOD)
        {
          PE::Checkbox("Use TwoPass Culling", (bool*)&m_rendererConfig.useTwoPassCulling,
                       "Use two pass culling in rasterization, otherwise uses only last frame's hiz");
          ImGui::BeginDisabled(!m_rendererConfig.useTwoPassCulling);
          PE::Checkbox("Adaptive TwoPass", &m_rendererConfig.useAdaptiveTwoPassCulling,
                       "Only run the second pass when the scene is large enough and culling matrices changed");
          PE::InputIntClamped("TwoPass min clusters", (int*)&m_rendererConfig.twoPassMinClusters, 0, 1 << 28, 256, 1024,
                              ImGuiInputTextFlags_EnterReturnsTrue);
          PE::InputFloat("TwoPass matrix delta", &m_rendererConfig.twoPassMatrixDelta, 0.0001f, 0.001f, "%.6f",
                         ImGuiInputTextFlags_EnterReturnsTrue);
          ImGui::EndDisabled();

          ImGui::EndDisabled();
        }
        else
        {
          PE::Checkbox("Force Invisible Culling", (bool*)&m_rendererConfig.useForcedInvisibleCulling,
                       "Even ray tracing will cull based on primary visibility alone. ");

          ImGui::EndDisabled();
        }

        PE::treePop();
      }

      PE::end();


      ImGui::Separator();

      if(m_rendererConfig.useRenderStats && ImGui::BeginTable("##Render stats", 3, ImGuiTableFlags_RowBg))
      {
        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 140.0f * ImGui::GetWindowDpiScale());

        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthStretch);

        ImGui::TableNextRow();

        ImGui::TableNextColumn();

        ImGui::Text("Enqueued Tasks");

        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMetric(readback.numTraversedTasks).c_str());

        ImGui::TableNextRow();

        ImGui::TableNextColumn();

        ImGui::Text("TwoPass Active");

        ImGui::TableNextColumn();
        ImGui::Text("%s", readback.twoPassCullingActive ? "yes" : "no");

        ImGui::TableNextRow();

        ImGui::TableNextColumn();

        ImGui::Text("Pass Tasks");

        ImGui::TableNextColumn();
        ImGui::Text("%s / %s", formatMetric(readback.numTraversedTasksPass0).c_str(),
                    formatMetric(readback.numTraversedTasksPass1).c_str());

        ImGui::TableNextRow();

        ImGui::TableNextColumn();

        ImGui::Text("Enqueued Clusters");

        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMetric(readback.numRenderedClusters).c_str());

        ImGui::TableNextRow();

        ImGui::TableNextColumn();

        ImGui::Text("Pass Clusters");

        ImGui::TableNextColumn();
        ImGui::Text("%s / %s", formatMetric(readback.numRenderedClustersPass0).c_str(),
                    formatMetric(readback.numRenderedClustersPass1).c_str());

        ImGui::TableNextRow();

        ImGui::TableNextColumn();

        ImGui::Text("Enqueued Triangles");

        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMetric(readback.numRenderedTriangles).c_str());

        ImGui::TableNextRow();

        ImGui::TableNextColumn();

        ImGui::Text("Enqueued Tri/Cluster");

        ImGui::TableNextColumn();
        if(readback.numRenderedClusters > 0)
        {
          ImGui::Text("%.1f", float(readback.numRenderedTriangles) / float(readback.numRenderedClusters));
        }
        else
        {

          ImGui::Text("N/A");
        }
        ImGui::EndTable();
      }
    }

    if(ImGui::CollapsingHeader("Clusters & LoDs generation", nullptr, ImGuiTreeNodeFlags_DefaultOpen))
    {

      ImGui::Text("Applying changes can take significant time");


      PE::begin("##Clusters", ImGuiTableFlags_Resizable);


      static int selectedTris = 128;
      static int selectedVerts = 128;


      if(selectedTris != m_sceneConfigEdit.clusterTriangles || selectedVerts != m_sceneConfigEdit.clusterVertices) {
        selectedTris = m_sceneConfigEdit.clusterTriangles;
        selectedVerts = m_sceneConfigEdit.clusterVertices;
      }


      PE::entry("Triangle Count", [&]() {
        const int trisOptions[] = {32, 64, 96, 128, 160, 192, 224, 256};
        if(ImGui::BeginCombo("##tris", std::to_string(selectedTris).c_str())) {
          for(int tris : trisOptions) {
            if(ImGui::Selectable(std::to_string(tris).c_str(), selectedTris == tris)) {
              selectedTris = tris;

              for(uint32_t i = 0; i < NUM_CLUSTER_CONFIGS; i++) {
                const ClusterInfo& entry = s_clusterInfos[i];
                if(entry.tris == selectedTris && entry.verts == selectedVerts) {
                  m_tweak.clusterConfig = entry.cfg;

                  setFromClusterConfig(m_sceneConfigEdit, m_tweak.clusterConfig);
                  break;
                }
              }
            }
          }

          ImGui::EndCombo();
        }
        return false;
      });


      PE::entry("Vertex Count", [&]() {
        const int vertsOptions[] = {32, 64, 96, 128, 160, 192, 224, 256};
        if(ImGui::BeginCombo("##verts", std::to_string(selectedVerts).c_str())) {
          for(int verts : vertsOptions) {
            if(ImGui::Selectable(std::to_string(verts).c_str(), selectedVerts == verts)) {
              selectedVerts = verts;

              for(uint32_t i = 0; i < NUM_CLUSTER_CONFIGS; i++) {
                const ClusterInfo& entry = s_clusterInfos[i];
                if(entry.tris == selectedTris && entry.verts == selectedVerts) {
                  m_tweak.clusterConfig = entry.cfg;

                  setFromClusterConfig(m_sceneConfigEdit, m_tweak.clusterConfig);
                  break;
                }
              }
            }
          }

          ImGui::EndCombo();
        }
        return false;
      });


      PE::Text("Current Config:", "%dT_%dV", m_sceneConfigEdit.clusterTriangles, m_sceneConfigEdit.clusterVertices);

      PE::entry("Semantic LOD Strategy", [&]() {
        return m_ui.enumCombobox(GUI_SEMANTIC_LOD_MODE, "semantic lod strategy",
                                 &m_sceneConfigEdit.semanticLodMode);
      }, "Selects which ablation policy CSV is loaded before rebuilding LOD data");


      if(PE::treeNode("Compression settings"))
      {

        PE::Checkbox("Enable compression", &m_sceneConfigEdit.useCompressedData, "Lowers cache file size, can speed up streaming");
        PE::InputIntClamped("POS Mantissa drop bits", (int*)&m_sceneConfigEdit.compressionPosDropBits, 0, 22, 1, 1,
                            ImGuiInputTextFlags_EnterReturnsTrue,
                            "position number of mantissa bits to drop (zeroed) to improve compression");
        PE::InputIntClamped("TC Mantissa drop bits", (int*)&m_sceneConfigEdit.compressionTexDropBits, 0, 22, 1, 1,
                            ImGuiInputTextFlags_EnterReturnsTrue,
                            "texcoord number of mantissa bits to drop (zeroed) to improve compression");

        PE::treePop();
      }


      bool hasChanges = memcmp(&m_sceneConfigEdit, &m_sceneConfig, sizeof(m_sceneConfigEdit)) != 0;


      ImGui::BeginDisabled(!hasChanges);
      if(hasChanges)
      {

        ImGui::PushStyleColor(ImGuiCol_Text, changesColor);
      }

      ImVec2 buttonSize = {100.0f * ImGui::GetWindowDpiScale(), 20 * ImGui::GetWindowDpiScale()};
      if(PE::entry("Operations", [&] { return ImGui::Button("Apply Changes", buttonSize); }, "Applying changes triggers reload and processing of the scene"))
      {
        m_sceneConfig = m_sceneConfigEdit;
      }
      if(hasChanges)
      {

        ImGui::PopStyleColor();
      }
      if(PE::entry("", [&] { return ImGui::Button("Reset Changes", buttonSize); }, "Resets the current edits"))
      {
        m_sceneConfigEdit     = m_sceneConfig;

        m_tweak.clusterConfig = findSceneClusterConfig(m_sceneConfig);
      }

      ImGui::EndDisabled();

      if(PE::entry("", [&] { return ImGui::Button("Reload Strategy", buttonSize); },
                   "Reloads the scene using the selected semantic LOD strategy"))
      {
        m_sceneConfig = m_sceneConfigEdit;
        m_reloadScene = true;
      }


      PE::end();
    }

    if(ImGui::CollapsingHeader("Model Array", nullptr, ImGuiTreeNodeFlags_DefaultOpen))
    {
      PE::begin("##ModelArray", ImGuiTableFlags_Resizable);

      PE::InputIntClamped("Array count", (int*)&m_sceneGridConfig.numCopies, int(SceneGridConfig::minCopies),
                          int(SceneGridConfig::maxCopies), 1, 1, ImGuiInputTextFlags_EnterReturnsTrue,
                          "Creates an instance array of the imported model without rebuilding LOD data");

      if(PE::treeNode("Advanced placement"))
      {
        PE::InputInt("Grid axes", (int*)&m_sceneGridConfig.gridBits, 1, 1, ImGuiInputTextFlags_EnterReturnsTrue,
                     "Bitmask: 1 X, 2 Y, 4 Z; default arranges copies on X/Z");
        PE::Checkbox("Unique geometry per copy", &m_sceneGridConfig.uniqueGeometriesForCopies,
                     "Duplicates geometry references per copy; normally leave disabled for array tests");
        PE::InputFloat("Spacing X", &m_sceneGridConfig.refShift.x, 0.1f, 0.1f, "%.2f", ImGuiInputTextFlags_EnterReturnsTrue);
        PE::InputFloat("Spacing Y", &m_sceneGridConfig.refShift.y, 0.1f, 0.1f, "%.2f", ImGuiInputTextFlags_EnterReturnsTrue);
        PE::InputFloat("Spacing Z", &m_sceneGridConfig.refShift.z, 0.1f, 0.1f, "%.2f", ImGuiInputTextFlags_EnterReturnsTrue);
        PE::treePop();
      }

      PE::end();
    }


   if(m_renderScene && ImGui::CollapsingHeader("Streaming", nullptr, ImGuiTreeNodeFlags_DefaultOpen))
   {

     PE::begin("##Streaming", ImGuiTableFlags_Resizable);
     if(m_renderSceneCanPreload)
     {

       ImGui::PushStyleColor(ImGuiCol_Text, recommendedColor);

       PE::Checkbox("Enable", &m_tweak.useStreaming);

       ImGui::PopStyleColor();
     }


     ImGui::BeginDisabled(m_renderScene == nullptr);
     if(PE::entry("Streaming state", [&] { return ImGui::Button("Reset"); }, "resets the streaming state"))
     {

       m_renderScene->streamingReset();
     }

     ImGui::EndDisabled();


     PE::InputIntClamped("Max Resident Groups", (int*)&m_streamingConfig.maxGroups,
                         uint32_t(m_scene ? m_scene->getActiveGeometryCount() : 1024 * 1024), 1024 * 1024, 128, 128,
                         ImGuiInputTextFlags_EnterReturnsTrue);

     PE::InputIntClamped("Max Geometry MiB", (int*)&m_streamingConfig.maxGeometryMegaBytes, 128, 1024 * 48, 128, 128,
                         ImGuiInputTextFlags_EnterReturnsTrue);

     if(PE::treeNode("Frame settings"))
     {
       PE::InputIntClamped("Unload frame delay", (int*)&m_frameConfig.streamingAgeThreshold, 2, 1024, 1, 1,
                           ImGuiInputTextFlags_EnterReturnsTrue);

       PE::InputIntClamped("Max Group Loads", (int*)&m_streamingConfig.maxPerFrameLoadRequests, 1, 16 * 1024 * 1024,
                           128, 128, ImGuiInputTextFlags_EnterReturnsTrue);
       PE::InputIntClamped("Max Group Unloads", (int*)&m_streamingConfig.maxPerFrameUnloadRequests, 1,
                           16 * 1024 * 1024, 128, 128, ImGuiInputTextFlags_EnterReturnsTrue);

       PE::InputIntClamped("Max Transfer MiB", (int*)&m_streamingConfig.maxTransferMegaBytes, 1, 1024, 1, 2,
                           ImGuiInputTextFlags_EnterReturnsTrue);


       PE::Checkbox("Async transfer", &m_streamingConfig.useAsyncTransfer, "Use asynchronous transfer queue for uploads");

       ImGui::BeginDisabled(!m_streamingConfig.useAsyncTransfer);
       PE::Checkbox("Decoupled transfer", &m_streamingConfig.useDecoupledAsyncTransfer,
                    "Allow asynchronous transfers to take multiple frames");

       ImGui::EndDisabled();

       PE::treePop();
     }


     PE::end();


     ImGui::Separator();

     if(ImGui::BeginTable("Streaming stats", 3, ImGuiTableFlags_RowBg))
     {
       ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 155.0f * ImGui::GetWindowDpiScale());

       ImGui::TableSetupColumn("Count", ImGuiTableColumnFlags_WidthStretch);

       ImGui::TableSetupColumn("Percentage", ImGuiTableColumnFlags_WidthStretch);

       ImGui::TableNextRow();

       ImGui::TableNextColumn();

       ImGui::Text("Geometry");

       ImGui::TableNextColumn();
       ImGui::TextColored(stats.couldNotStore ? warn_color : text_color, "%s", formatMemorySize(stats.usedDataBytes).c_str());

       ImGui::TableNextColumn();
       ImGui::TextColored(stats.couldNotStore ? warn_color : text_color, "%d %%",
                          getUsagePct(stats.usedDataBytes, stats.maxDataBytes));

       ImGui::TableNextRow();

       ImGui::TableNextColumn();

       ImGui::Text("Resident groups");

       ImGui::TableNextColumn();
       ImGui::TextColored(stats.couldNotAllocateGroup ? warn_color : text_color, "%s",
                          formatMetric(stats.residentGroups).c_str());

       ImGui::TableNextColumn();
       ImGui::TextColored(stats.couldNotAllocateGroup ? warn_color : text_color, "%d %%",
                          getUsagePct(stats.residentGroups, stats.maxGroups));

       ImGui::TableNextRow();

       ImGui::TableNextColumn();


       ImGui::Text("Resident clusters");

       uint32_t pctClusters = getUsagePct(stats.residentClusters, stats.maxClusters);

       ImGui::TableNextColumn();
       ImGui::TextColored(pctClusters > 99 ? warn_color : text_color, "%s", formatMetric(stats.residentClusters).c_str());

       ImGui::TableNextColumn();

       ImGui::TextColored(pctClusters > 99 ? warn_color : text_color, "%d %%", pctClusters);

       ImGui::TableNextRow();

       ImGui::TableNextColumn();


       ImGui::Text("Last Completed Transfer");

       ImGui::TableNextColumn();
       ImGui::TextColored(stats.couldNotTransfer ? warn_color : text_color, "%s", formatMemorySize(stats.transferBytes).c_str());

       ImGui::TableNextColumn();
       ImGui::TextColored(stats.couldNotTransfer ? warn_color : text_color, "%d %%",
                          getUsagePct(stats.transferBytes, stats.maxTransferBytes));

       ImGui::TableNextRow();

       ImGui::TableNextColumn();
#if 0

     ImGui::Text("Last Completed Transfers");

     ImGui::TableNextColumn();
     ImGui::TextColored(text_color, "%s", formatMetric(stats.transferCount).c_str());

     ImGui::TableNextColumn();

     ImGui::TextColored(text_color, "-");

     ImGui::TableNextRow();

     ImGui::TableNextColumn();
#endif

       uint32_t pctLoad =
           stats.loadCount == m_streamingConfig.maxPerFrameLoadRequests ?
               100 :
               std::min(99u, uint32_t(float(stats.loadCount) * 100.0f / float(m_streamingConfig.maxPerFrameLoadRequests)));


       ImGui::Text("Last Completed Loads");

       ImGui::TableNextColumn();
       ImGui::TextColored(pctLoad == 100 ? warn_color : text_color, "%s", formatMetric(stats.loadCount).c_str());

       ImGui::TableNextColumn();

       ImGui::TextColored(pctLoad == 100 ? warn_color : text_color, "%d %%", pctLoad);

       ImGui::TableNextRow();

       ImGui::TableNextColumn();

       uint32_t pctUnLoad =
           stats.unloadCount == m_streamingConfig.maxPerFrameUnloadRequests ?
               100 :
               std::min(99u, uint32_t(float(stats.unloadCount) * 100.0f / float(m_streamingConfig.maxPerFrameUnloadRequests)));


       ImGui::Text("Last Completed Unloads");

       ImGui::TableNextColumn();
       ImGui::TextColored(pctUnLoad == 100 ? warn_color : text_color, "%s", formatMetric(stats.unloadCount).c_str());

       ImGui::TableNextColumn();

       ImGui::TextColored(pctUnLoad == 100 ? warn_color : text_color, "%d %%", pctUnLoad);

       ImGui::TableNextRow();

       ImGui::TableNextColumn();

       uint32_t pctUncompleted =
           stats.uncompletedLoadCount == m_streamingConfig.maxPerFrameLoadRequests ?
               100 :
               std::min(99u, uint32_t(float(stats.unloadCount) * 100.0f / float(m_streamingConfig.maxPerFrameUnloadRequests)));


       ImGui::Text("Last Uncompleted Loads");

       ImGui::TableNextColumn();
       ImGui::TextColored(stats.uncompletedLoadCount ? warn_color : text_color, "%s",
                          formatMetric(stats.uncompletedLoadCount).c_str());

       ImGui::TableNextColumn();

       ImGui::TextColored(stats.uncompletedLoadCount ? warn_color : text_color, "%d %%", pctUncompleted);

       ImGui::TableNextRow();

       ImGui::TableNextColumn();


       ImGui::EndTable();
     }
   }

  }

  ImGui::End();


  Renderer::ResourceUsageInfo resourceActual = m_renderer ? m_renderer->getResourceUsage(false) : Renderer::ResourceUsageInfo();

  Renderer::ResourceUsageInfo resourceReserved = m_renderer ? m_renderer->getResourceUsage(true) : Renderer::ResourceUsageInfo();

  if(ImGui::Begin("Streaming memory"))
  {
    const uint32_t maxSlots = 512;
    if(m_streamGeometryHistogram.empty() == m_tweak.useStreaming)
    {
      m_streamGeometryHistogramMax = 0;
      m_streamHistogramOffset      = 0;

      m_streamGeometryHistogram.resize(m_tweak.useStreaming ? maxSlots : 0, 0);
    }

    if(m_renderScene && !m_streamGeometryHistogram.empty())
    {

      m_renderScene->sceneStreaming.getStats(stats);

#if MEMORY_WITH_BINARY_PREFIXES
      size_t divisor = 1024 * 1024;


#define MEMORY_MB "MiB"
#else
      size_t divisor = 1000000;


#define MEMORY_MB "MB"
#endif


      uint32_t mbGeometry          = uint32_t((stats.usedDataBytes + divisor - 1) / divisor);

      m_streamGeometryHistogramMax = std::max(m_streamGeometryHistogramMax, mbGeometry);
      {
        m_streamHistogramOffset = (m_streamHistogramOffset + 1) % maxSlots;
        m_streamGeometryHistogram[(m_streamHistogramOffset + maxSlots - 1) % maxSlots] = mbGeometry;
      }
      uiPlot(std::string("Streaming Geometry Memory (" MEMORY_MB ")"), std::string("past %d " MEMORY_MB " %d"),
             m_streamGeometryHistogram, m_streamGeometryHistogramMax, m_streamHistogramOffset);
    }
  }

  ImGui::End();

  if(ImGui::Begin("Statistics"))
  {
    if(m_scene && ImGui::CollapsingHeader("Scene", nullptr, ImGuiTreeNodeFlags_DefaultOpen))
    {
      if(ImGui::BeginTable("Scene stats", 3, ImGuiTableFlags_None))
      {

        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthStretch);

        ImGui::TableSetupColumn("Scene", ImGuiTableColumnFlags_WidthStretch);

        ImGui::TableSetupColumn("Model", ImGuiTableColumnFlags_WidthStretch);

        ImGui::TableHeadersRow();

        ImGui::TableNextRow();

        ImGui::TableNextColumn();

        ImGui::Text("Triangles");

        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMetric(m_scene->m_hiTrianglesCountInstanced * m_sceneGridConfig.numCopies).c_str());

        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMetric(m_scene->m_hiTrianglesCount).c_str());

        ImGui::TableNextRow();

        ImGui::TableNextColumn();

        ImGui::Text("Clusters");

        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMetric(m_scene->m_hiClustersCountInstanced * m_sceneGridConfig.numCopies).c_str());

        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMetric(m_scene->m_hiClustersCount).c_str());

        ImGui::TableNextRow();

        ImGui::TableNextColumn();

        ImGui::Text("Instances");

        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMetric(m_scene->m_instances.size()).c_str());

        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMetric(m_scene->m_originalInstanceCount).c_str());

        ImGui::TableNextRow();

        ImGui::TableNextColumn();

        ImGui::Text("Geometries");

        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMetric(m_scene->getActiveGeometryCount()).c_str());

        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMetric(m_scene->m_originalGeometryCount).c_str());

        ImGui::EndTable();
      }
    }
    if(m_renderer && ImGui::CollapsingHeader("Traversal", nullptr, ImGuiTreeNodeFlags_DefaultOpen))
    {
      if(ImGui::BeginTable("Traversal stats", 3, ImGuiTableFlags_RowBg))
      {

        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthStretch);

        ImGui::TableSetupColumn("Requested", ImGuiTableColumnFlags_WidthStretch);

        ImGui::TableSetupColumn("Reserved", ImGuiTableColumnFlags_WidthStretch);

        ImGui::TableHeadersRow();

        ImGui::TableNextRow();

        ImGui::TableNextColumn();

        ImGui::Text("Tasks");

        ImGui::TableNextColumn();
        ImGui::TextColored(pct.pctTasks > 100 ? warn_color : text_color, "%s", formatMetric(readback.numTraversalTasks).c_str());

        ImGui::TableNextColumn();

        ImGui::TextColored(pct.pctTasks > 100 ? warn_color : text_color, "%d %%", pct.pctTasks);

        ImGui::TableNextRow();

        ImGui::TableNextColumn();

        ImGui::Text("Clusters");

        ImGui::TableNextColumn();
        ImGui::TextColored(pct.pctClusters > 100 ? warn_color : text_color, "%s",
                           formatMetric(readback.numRenderClusters).c_str());

        ImGui::TableNextColumn();

        ImGui::TextColored(pct.pctClusters > 100 ? warn_color : text_color, "%d %%", pct.pctClusters);

        ImGui::TableNextRow();

        ImGui::TableNextColumn();

        ImGui::EndTable();
      }
    }
    if(m_renderer && ImGui::CollapsingHeader("Memory", nullptr, ImGuiTreeNodeFlags_DefaultOpen))
    {
      if(ImGui::BeginTable("Memory stats", 3, ImGuiTableFlags_RowBg))
      {

        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthStretch);

        ImGui::TableSetupColumn("Actual", ImGuiTableColumnFlags_WidthStretch);

        ImGui::TableSetupColumn("Reserved", ImGuiTableColumnFlags_WidthStretch);

        ImGui::TableHeadersRow();

        ImGui::TableNextRow();

        ImGui::TableNextColumn();

        ImGui::Text("Geometry");

        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(resourceActual.geometryMemBytes).c_str());

        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(resourceReserved.geometryMemBytes).c_str());

        ImGui::TableNextRow();

        ImGui::TableNextColumn();

        ImGui::Text("Operations");

        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(resourceActual.operationsMemBytes).c_str());

        ImGui::TableNextColumn();

        ImGui::Text("==");

        ImGui::TableNextRow();

        ImGui::TableNextColumn();

        ImGui::Text("Total");

        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(resourceActual.getTotalSum()).c_str());

        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(resourceReserved.getTotalSum()).c_str());

        ImGui::TableNextRow();

        ImGui::TableNextColumn();

        ImGui::EndTable();
      }
    }

    if(m_scene && ImGui::CollapsingHeader("Model Cluster Stats"))
    {

      ImGui::Text("Cluster max triangles: %d", m_scene->m_maxClusterTriangles);

      ImGui::Text("Cluster max vertices: %d", m_scene->m_maxClusterVertices);

      ImGui::Text("Cluster count: %" PRIu64, m_scene->m_totalClustersCount);
      ImGui::Text("Clusters with config (%u) triangles: %u (%.1f%%)", m_scene->m_config.clusterTriangles,
                  m_scene->m_histograms.clusterTriangles[m_scene->m_config.clusterTriangles],
                  float(m_scene->m_histograms.clusterTriangles[m_scene->m_config.clusterTriangles]) * 100.f
                      / float(m_scene->m_totalClustersCount));

      ImGui::Text("Geometry max lod levels: %d", m_scene->m_maxLodLevelsCount);

      uiPlot(std::string("Cluster Triangles Histogram"), std::string("Cluster count with %d triangles: %u"),
             m_scene->m_histograms.clusterTriangles, m_scene->m_histograms.clusterTrianglesMax, 0,
             m_scene->m_config.clusterTriangles + 1);

      uiPlot(std::string("Cluster Vertices Histogram"), std::string("Cluster count with %d vertices: %u"),
             m_scene->m_histograms.clusterVertices, m_scene->m_histograms.clusterVerticesMax, 0,
             m_scene->m_config.clusterVertices + 1);

      uiPlot(std::string("Group Clusters Histogram"), std::string("Group count with %d clusters: %u"),
             m_scene->m_histograms.groupClusters, m_scene->m_histograms.groupClustersMax, 0,
             m_scene->m_config.clusterGroupSize + 1);

      uiPlot(std::string("Node Children Histogram"), std::string("Node count with %d children: %u"),
             m_scene->m_histograms.nodeChildren, m_scene->m_histograms.nodeChildrenMax);

      uiPlot(std::string("LOD Levels Histogram"), std::string("Mesh count with %d LOD levels: %u"),
             m_scene->m_histograms.lodLevels, m_scene->m_histograms.lodLevelsMax, 0, m_scene->m_maxLodLevelsCount + 1);
    }
  }

  ImGui::End();

  if(ImGui::Begin("Misc Settings"))
  {
    if(ImGui::CollapsingHeader("Camera", nullptr, ImGuiTreeNodeFlags_DefaultOpen))
    {

      nvgui::CameraWidget(m_info.cameraManipulator, false);
      namespace PE = nvgui::PropertyEditor;

      PE::begin("misc", ImGuiTableFlags_Resizable);
      PE::InputFloat("Speed distance factor", &m_tweak.clickSpeedScale, 0, 0, "%.2f", 0,
                     "double click causes speed to be based on this percentage of the distance to hit point");

      PE::end();
    }

    if(ImGui::CollapsingHeader("Lighting", nullptr, ImGuiTreeNodeFlags_DefaultOpen))
    {
      namespace PE = nvgui::PropertyEditor;

      PE::begin("misc", ImGuiTableFlags_Resizable);
      PE::SliderFloat("Light Mixer", &m_frameConfig.frameConstants.lightMixer, 0.0f, 1.0f, "%.3f", 0,
                      "Mix between flashlight and sun light");

      PE::end();

      ImGui::Text("Sun & Sky");

      nvgui::skySimpleParametersUI(m_frameConfig.frameConstants.skyParams, "misc", ImGuiTableFlags_Resizable);
    }


    if(ImGui::CollapsingHeader("Advanced", nullptr, ImGuiTreeNodeFlags_DefaultOpen))
    {

      PE::begin("misc", ImGuiTableFlags_Resizable);
      PE::InputIntClamped("Persistent traversal threads", (int*)&m_frameConfig.traversalPersistentThreads, 32,
                          256 * 1024, 1, 1, ImGuiInputTextFlags_EnterReturnsTrue);
      PE::InputInt("Colorize xor", (int*)&m_frameConfig.frameConstants.colorXor);

      PE::Checkbox("Auto reset timer", &m_tweak.autoResetTimers);
      if(m_resources.m_supportsMeshShaderNV)
      {

        PE::Checkbox("Use EXT Mesh shader", &m_rendererConfig.useEXTmeshShader);

      }

      PE::end();
    }
  }

  ImGui::End();

  if(m_showDebugUI)
  {
    if(ImGui::Begin("Debug"))
    {
      if(ImGui::CollapsingHeader("Debug Shader Values", nullptr, ImGuiTreeNodeFlags_DefaultOpen))
      {

        PE::begin("##HiddenID");
        PE::InputInt("dbgInt", (int*)&m_frameConfig.frameConstants.dbgUint, 1, 100, ImGuiInputTextFlags_EnterReturnsTrue);

        PE::InputFloat("dbgFloat", &m_frameConfig.frameConstants.dbgFloat, 0.1f, 1.0f, "%.3f", ImGuiInputTextFlags_EnterReturnsTrue);

        PE::end();


        ImGui::Text(" debugI :  %10d", readback.debugI);

        ImGui::Text(" debugUI:  %10u", readback.debugUI);

        ImGui::Text(" debugU64:  %" PRIX64, readback.debugU64);
        static bool debugFloat = false;
        static bool debugHex   = false;
        static bool debugAll   = false;

        ImGui::Checkbox(" as float", &debugFloat);

        ImGui::SameLine();

        ImGui::Checkbox("hex", &debugHex);

        ImGui::SameLine();

        ImGui::Checkbox("all", &debugAll);


        uint32_t count = debugAll ? 64 : 32;

        if(ImGui::BeginTable("##Debug", 4, ImGuiTableFlags_BordersOuter))
        {

          ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 32);

          ImGui::TableSetupColumn("A", ImGuiTableColumnFlags_WidthStretch);

          ImGui::TableSetupColumn("B", ImGuiTableColumnFlags_WidthStretch);

          ImGui::TableSetupColumn("C", ImGuiTableColumnFlags_WidthStretch);

          ImGui::TableHeadersRow();

          ImGui::TableNextRow();

          ImGui::TableNextColumn();
          for(uint32_t i = 0; i < count; i++)
          {

            ImGui::Text("%2d", i);
            if(debugFloat)
            {

              ImGui::TableNextColumn();
              ImGui::Text("%f", *(float*)&readback.debugA[i]);

              ImGui::TableNextColumn();
              ImGui::Text("%f", *(float*)&readback.debugB[i]);

              ImGui::TableNextColumn();
              ImGui::Text("%f", *(float*)&readback.debugC[i]);
            }
            else if(debugHex)
            {

              ImGui::TableNextColumn();

              ImGui::Text("%X", readback.debugA[i]);

              ImGui::TableNextColumn();

              ImGui::Text("%X", readback.debugB[i]);

              ImGui::TableNextColumn();

              ImGui::Text("%X", readback.debugC[i]);
            }
            else
            {

              ImGui::TableNextColumn();

              ImGui::Text("%d", readback.debugA[i]);

              ImGui::TableNextColumn();

              ImGui::Text("%d", readback.debugB[i]);

              ImGui::TableNextColumn();

              ImGui::Text("%d", readback.debugC[i]);
            }


            ImGui::TableNextRow();

            ImGui::TableNextColumn();
          }


          ImGui::EndTable();
        }
      }
    }

    ImGui::End();
  }


  handleChanges();


  if(ImGui::Begin("Viewport"))
  {

    ImVec2 corner = ImGui::GetCursorScreenPos();
    ImGui::Image((ImTextureID)m_imguiTexture, ImGui::GetContentRegionAvail());

    viewportUI(corner);
  }

  ImGui::End();

  {
    const ImGuiViewport* mainViewport = ImGui::GetMainViewport();
    const ImVec2         margin(12.0f, 12.0f);
    const ImVec2         panelSize(std::min(560.0f, mainViewport->WorkSize.x * 0.42f),
                           std::min(720.0f, mainViewport->WorkSize.y * 0.62f));
    ImGui::SetNextWindowPos(ImVec2(mainViewport->WorkPos.x + mainViewport->WorkSize.x - margin.x,
                                   mainViewport->WorkPos.y + mainViewport->WorkSize.y - margin.y),
                            ImGuiCond_Always, ImVec2(1.0f, 1.0f));
    ImGui::SetNextWindowSize(panelSize, ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.88f);

    ImGuiWindowFlags flags = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings;
    if(ImGui::Begin("Runtime / Cache Parameters###BottomRightRuntimeParameters", nullptr, flags))
    {
      auto rowText = [](const char* name, const std::string& value) {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(name);
        ImGui::TableNextColumn();
        ImGui::TextWrapped("%s", value.c_str());
      };

      auto rowFmt = [&](const char* name, const char* format, auto... args) {
        rowText(name, fmt::format(fmt::runtime(format), args...));
      };

      auto rowBool = [&](const char* name, bool value) { rowText(name, value ? "true" : "false"); };

      auto beginParamTable = [](const char* id) {
        return ImGui::BeginTable(id, 2, ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_SizingStretchProp);
      };

      ImGui::Text("Frame %d", m_frames);
      if(m_sceneLoading)
      {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.95f, 0.65f, 0.15f, 1.0f), "Loading %u%%", uint32_t(m_sceneProgress.load()));
      }

      if(ImGui::BeginChild("##BottomRightRuntimeScroll", ImVec2(0, 0), false, ImGuiWindowFlags_AlwaysVerticalScrollbar))
      {
        if(ImGui::CollapsingHeader("Scene / Cache", ImGuiTreeNodeFlags_DefaultOpen))
        {
          if(beginParamTable("##SceneCacheParams"))
          {
            rowText("Scene file", m_sceneFilePath.empty() ? std::string("<none>") : m_sceneFilePath.string());
            rowText("Pending file", m_sceneFilePathDropNew.empty() ? std::string("<none>") : m_sceneFilePathDropNew.string());
            rowText("Cache suffix", m_sceneCacheSuffix);
            if(m_scene)
            {
              const std::filesystem::path& cachePath = m_scene->getCacheFilePath();
              const bool cacheExists = std::filesystem::exists(cachePath);
              rowText("Cache file", cachePath.string());
              rowBool("Cache exists", cacheExists);
              rowBool("Loaded from cache", m_scene->m_loadedFromCache);
              rowBool("Memory mapped cache", m_scene->isMemoryMappedCache());
              rowFmt("Cache size", "{}", cacheExists ? formatMemorySize(size_t(std::filesystem::file_size(cachePath))) : "0 B");
              rowFmt("Assembly nodes", "{}", m_scene->m_assemblyNodes.size());
              rowFmt("Assembly templates", "{}", m_scene->m_assemblyTemplates.size());
            }
            else
            {
              rowText("Cache file", "<no scene>");
            }
            rowBool("Auto load cache", m_sceneLoaderConfig.autoLoadCache);
            rowBool("Auto save cache", m_sceneLoaderConfig.autoSaveCache);
            rowBool("Processing only", m_sceneLoaderConfig.processingOnly);
            rowBool("Processing partial", m_sceneLoaderConfig.processingAllowPartial);
            rowFmt("Processing threads", "{:.2f}", m_sceneLoaderConfig.processingThreadsPct);
            ImGui::EndTable();
          }
        }

        if(ImGui::CollapsingHeader("Scene Config", ImGuiTreeNodeFlags_DefaultOpen))
        {
          const SceneConfig& cfg = m_scene ? m_scene->m_config : m_sceneConfig;
          if(beginParamTable("##SceneConfigParams"))
          {
            rowFmt("Config version", "{}", SceneConfig::version);
            rowFmt("Cluster vertices", "{}", cfg.clusterVertices);
            rowFmt("Cluster triangles", "{}", cfg.clusterTriangles);
            rowFmt("Cluster group size", "{}", cfg.clusterGroupSize);
            rowFmt("LOD node width", "{}", cfg.preferredNodeWidth);
            rowFmt("LOD decimation", "{:.3f}", cfg.lodLevelDecimationFactor);
            rowFmt("Assembly min instances", "{}", cfg.assemblyCullingMinInstances);
            rowFmt("Assembly LOD pixels", "{:.2f}", cfg.assemblyLodPixelThreshold);
            rowText("Semantic LOD mode", semanticLodModeName(cfg.semanticLodMode));
            rowBool("Feature constraints", cfg.featureConstraints);
            rowFmt("Feature importance weight", "{:.2f}", cfg.featureImportanceWeight);
            rowFmt("Feature protect threshold", "{:.2f}", cfg.featureProtectThreshold);
            rowFmt("Feature critical threshold", "{:.2f}", cfg.featureCriticalThreshold);
            rowFmt("Meshopt fill weight", "{:.3f}", cfg.meshoptFillWeight);
            rowFmt("Meshopt split factor", "{:.3f}", cfg.meshoptSplitFactor);
            rowBool("Compressed data", cfg.useCompressedData);
            rowFmt("Compressed pos bits", "{}", cfg.compressionPosDropBits);
            rowFmt("Compressed tex bits", "{}", cfg.compressionTexDropBits);
            rowFmt("Enabled attributes", "0x{:X}", cfg.enabledAttributes);
            ImGui::EndTable();
          }
        }

        if(ImGui::CollapsingHeader("Runtime Renderer", ImGuiTreeNodeFlags_DefaultOpen))
        {
          if(beginParamTable("##RuntimeRendererParams"))
          {
            rowFmt("LOD pixel error", "{:.3f}", m_frameConfig.lodPixelError);
            rowFmt("Render resolution", "{} x {}", m_resources.m_frameBuffer.renderSize.width, m_resources.m_frameBuffer.renderSize.height);
            rowFmt("Visualize", "{}", uint32_t(m_frameConfig.visualize));
            rowBool("Freeze culling", m_frameConfig.freezeCulling);
            rowBool("Freeze LOD", m_frameConfig.freezeLoD);
            rowBool("Culling", m_rendererConfig.useCulling);
            rowBool("Two-pass culling", m_rendererConfig.useTwoPassCulling);
            rowBool("Adaptive two-pass", m_rendererConfig.useAdaptiveTwoPassCulling);
            rowFmt("Two-pass min clusters", "{}", m_rendererConfig.twoPassMinClusters);
            rowFmt("Two-pass matrix delta", "{:.6f}", m_rendererConfig.twoPassMatrixDelta);
            rowBool("Separate groups", m_rendererConfig.useSeparateGroups);
            rowBool("Instance sorting", m_rendererConfig.useSorting);
            rowBool("Render stats", m_rendererConfig.useRenderStats);
            ImGui::EndTable();
          }
        }

        if(m_scene && ImGui::CollapsingHeader("Scene Output Stats", ImGuiTreeNodeFlags_DefaultOpen))
        {
          if(beginParamTable("##SceneOutputStats"))
          {
            rowFmt("Geometries", "{} / {}", m_scene->getActiveGeometryCount(), m_scene->m_originalGeometryCount);
            rowFmt("Instances", "{} / {}", m_scene->m_instances.size(), m_scene->m_originalInstanceCount);
            rowFmt("HI clusters", "{}", m_scene->m_hiClustersCount);
            rowFmt("HI triangles", "{}", m_scene->m_hiTrianglesCount);
            rowFmt("HI vertices", "{}", m_scene->m_hiVerticesCount);
            rowFmt("Total clusters", "{}", m_scene->m_totalClustersCount);
            rowFmt("Total triangles", "{}", m_scene->m_totalTrianglesCount);
            rowFmt("Total vertices", "{}", m_scene->m_totalVerticesCount);
            rowFmt("Max geometry LOD levels", "{}", m_scene->m_maxLodLevelsCount);
            rowFmt("Max cluster triangles", "{}", m_scene->m_maxClusterTriangles);
            rowFmt("Max cluster vertices", "{}", m_scene->m_maxClusterVertices);
            ImGui::EndTable();
          }
        }

        if(m_scene && ImGui::CollapsingHeader("Semantic LOD Policy Distribution", ImGuiTreeNodeFlags_DefaultOpen))
        {
          static const char* policyNames[] = {
              "",
              "P1 micro / uncertain",
              "P2 repeated fastener",
              "P3 large static bulk",
              "P4 ordinary low detail",
              "P5 balanced visible",
              "P6 high-detail shape",
              "P7 interface / fluid",
              "P8 structural / control",
              "P9 motion / precision",
              "P10 critical preserve",
          };

          std::array<uint64_t, 11> instanceCounts = {};
          std::array<std::unordered_set<uint32_t>, 11> geometrySets;
          uint64_t semanticInstances = 0;

          for(const Scene::Instance& instance : m_scene->m_instances)
          {
            const uint32_t flags = instance.lodPolicy;
            if((flags & SEMANTIC_LOD_VALID_BIT) == 0)
              continue;

            const uint32_t priority = (flags >> SEMANTIC_LOD_PRIORITY_SHIFT) & SEMANTIC_LOD_PRIORITY_MASK;
            if(priority == 0 || priority >= instanceCounts.size())
              continue;

            instanceCounts[priority]++;
            geometrySets[priority].insert(instance.geometryID);
            semanticInstances++;
          }

          if(semanticInstances == 0)
          {
            ImGui::TextWrapped("No semantic LOD policy is attached to the current scene. Generate and load *_lod_constraints_fused.csv or *_lod_constraints.csv to populate this table.");
          }
          else if(ImGui::BeginTable("##SemanticLodPolicyDistribution", 4, ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV))
          {
            ImGui::TableSetupColumn("Policy", ImGuiTableColumnFlags_WidthFixed, 155.0f * ImGui::GetWindowDpiScale());
            ImGui::TableSetupColumn("Instances", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("Geometries", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("Share", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableHeadersRow();

            for(uint32_t priority = 1; priority <= 10; ++priority)
            {
              const uint64_t instances = instanceCounts[priority];
              const uint64_t geometries = uint64_t(geometrySets[priority].size());
              const double share = 100.0 * double(instances) / double(std::max<uint64_t>(1, semanticInstances));

              ImGui::TableNextRow();
              ImGui::TableNextColumn();
              ImGui::TextUnformatted(policyNames[priority]);
              ImGui::TableNextColumn();
              ImGui::Text("%s", formatMetric(instances).c_str());
              ImGui::TableNextColumn();
              ImGui::Text("%s", formatMetric(geometries).c_str());
              ImGui::TableNextColumn();
              ImGui::Text("%.2f %%", share);
            }

            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted("Total semantic");
            ImGui::TableNextColumn();
            ImGui::Text("%s", formatMetric(semanticInstances).c_str());
            ImGui::TableNextColumn();
            ImGui::Text("-");
            ImGui::TableNextColumn();
            ImGui::Text("100.00 %%");
            ImGui::EndTable();
          }
        }

        if(m_scene && ImGui::CollapsingHeader("Cache Generation Output", ImGuiTreeNodeFlags_DefaultOpen))
        {
          const Scene::ProcessingStatsSnapshot& ps = m_scene->m_processingStats;
          if(ps.groups == 0 && m_scene->m_loadedFromCache)
          {
            ImGui::TextWrapped("Generation stats are not stored in existing cache. Rebuild cache in this session to populate this section.");
          }
          if(beginParamTable("##ProcessingOutputStats"))
          {
            rowFmt("Groups", "{}", ps.groups);
            rowFmt("Clusters", "{}", ps.clusters);
            rowFmt("Vertices", "{}", ps.vertices);
            rowFmt("Group unique verts", "{}", ps.groupUniqueVertices);
            rowFmt("Group header bytes", "{}", formatMemorySize(size_t(ps.groupHeaderBytes)));
            rowFmt("Cluster header bytes", "{}", formatMemorySize(size_t(ps.clusterHeaderBytes)));
            rowFmt("Cluster bbox bytes", "{}", formatMemorySize(size_t(ps.clusterBboxBytes)));
            rowFmt("Cluster gen bytes", "{}", formatMemorySize(size_t(ps.clusterGenBytes)));
            rowFmt("Triangle index bytes", "{}", formatMemorySize(size_t(ps.triangleIndexBytes)));
            rowFmt("Vertex all bytes", "{}", formatMemorySize(size_t(ps.vertexPosBytes + ps.vertexTexCoordBytes + ps.vertexNrmBytes)));
            rowFmt("Vertex pos bytes", "{}", formatMemorySize(size_t(ps.vertexPosBytes)));
            rowFmt("Vertex tex bytes", "{}", formatMemorySize(size_t(ps.vertexTexCoordBytes)));
            rowFmt("Vertex normal/tangent bytes", "{}", formatMemorySize(size_t(ps.vertexNrmBytes)));
            rowFmt("Vertex compressed bytes", "{}", formatMemorySize(size_t(ps.vertexCompressedBytes)));
            ImGui::EndTable();
          }
        }

        if(m_scene && ImGui::CollapsingHeader("Feature Retention Output", ImGuiTreeNodeFlags_DefaultOpen))
        {
          const Scene::ProcessingStatsSnapshot& ps = m_scene->m_processingStats;
          double invVertices = 100.0 / double(std::max<uint64_t>(1, ps.inputFeatureVertices));
          double avgImportance = double(ps.featureImportanceSumPpm) / double(std::max<uint64_t>(1, ps.inputFeatureVertices)) / 1000000.0;
          double maxImportance = double(ps.featureImportanceMaxPpm) / 1000000.0;
          if(beginParamTable("##FeatureRetentionOutput"))
          {
            rowFmt("Input feature vertices", "{}", ps.inputFeatureVertices);
            rowFmt("Input feature tris", "{}", ps.inputFeatureTris);
            rowFmt("Boundary vertices", "{} ({:.2f}%)", ps.boundaryVertices, double(ps.boundaryVertices) * invVertices);
            rowFmt("Non-manifold vertices", "{} ({:.2f}%)", ps.nonManifoldVertices, double(ps.nonManifoldVertices) * invVertices);
            rowFmt("Sharp edge vertices", "{} ({:.2f}%)", ps.sharpEdgeVertices, double(ps.sharpEdgeVertices) * invVertices);
            rowFmt("Boundary components", "{}", ps.boundaryComponents);
            rowFmt("Sharp ring components", "{}", ps.sharpRingComponents);
            rowFmt("Circular hole loops", "{}", ps.circularHoleLoops);
            rowFmt("Circular hole vertices", "{} ({:.2f}%)", ps.circularHoleVertices, double(ps.circularHoleVertices) * invVertices);
            rowFmt("Functional boundaries", "{} ({:.2f}%)", ps.functionalBoundaryVertices, double(ps.functionalBoundaryVertices) * invVertices);
            rowFmt("Cylindrical vertices", "{} ({:.2f}%)", ps.cylindricalVertices, double(ps.cylindricalVertices) * invVertices);
            rowFmt("Thin-wall vertices", "{} ({:.2f}%)", ps.thinWallVertices, double(ps.thinWallVertices) * invVertices);
            rowFmt("Protected vertices", "{} ({:.2f}%)", ps.protectedVertices, double(ps.protectedVertices) * invVertices);
            rowFmt("Critical vertices", "{} ({:.2f}%)", ps.criticalVertices, double(ps.criticalVertices) * invVertices);
            rowFmt("Avg feature importance", "{:.4f}", avgImportance);
            rowFmt("Max feature importance", "{:.4f}", maxImportance);
            ImGui::EndTable();
          }
        }

        if(ImGui::CollapsingHeader("Streaming / Grid"))
        {
          if(beginParamTable("##StreamingGridParams"))
          {
            rowFmt("Grid copies", "{}", m_sceneGridConfig.numCopies);
            rowFmt("Grid config", "{}", m_sceneGridConfig.gridBits);
            rowBool("Grid unique geometry", m_sceneGridConfig.uniqueGeometriesForCopies);
            rowBool("Use streaming", m_tweak.useStreaming);
            rowFmt("Max resident groups", "{}", m_streamingConfig.maxGroups);
            rowFmt("Max geometry memory", "{}", formatMemorySize(m_streamingConfig.maxGeometryMegaBytes * 1024ull * 1024ull));
            if(m_renderScene && m_renderScene->useStreaming)
            {
              rowFmt("Resident groups", "{}", stats.residentGroups);
              rowFmt("Used geometry memory", "{}", formatMemorySize(size_t(stats.usedDataBytes)));
            }
            ImGui::EndTable();
          }
        }
      }
      ImGui::EndChild();
    }
    ImGui::End();
  }
}


void LodClusters::onUIMenu()
{

  bool vsync = m_app->isVsync();

  bool doOpenFile        = ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_O);

  bool doSaveCacheFile   = ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_S);

  bool doReloadFile      = ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_R);

  bool doDeleteCacheFile = ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_D);

  bool doCloseApp        = ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_Q);

  bool doToggleVsync     = ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_V);
  bool hasCache = m_scene && !m_scene->isMemoryMappedCache() && std::filesystem::exists(m_scene->getCacheFilePath());


  if(ImGui::BeginMenu("File"))
  {
    if(ImGui::MenuItem(ICON_MS_FILE_OPEN "Open", "Ctrl+O"))
    {
      doOpenFile = true;
    }
    if(m_scene)
    {
      if(ImGui::MenuItem(ICON_MS_REFRESH "Reload File", "Ctrl+R"))
      {
        doReloadFile = true;
      }

      if(!m_scene->m_loadedFromCache)
      {
        if(ImGui::MenuItem(ICON_MS_FILE_SAVE "Save Cache", "Ctrl+S"))
        {
          doSaveCacheFile = true;
        }
      }

      if(hasCache)
      {
        if(ImGui::MenuItem(ICON_MS_DELETE "Delete Cache", "Ctrl+D"))
        {
          doDeleteCacheFile = true;
        }
      }
    }
    if(ImGui::MenuItem(ICON_MS_DIRECTORY_SYNC "Reload Shaders", "R"))
    {
      m_reloadShaders = true;
    }
    if(ImGui::MenuItem(ICON_MS_POWER_SETTINGS_NEW "Exit", "Ctrl+Q"))
    {
      doCloseApp = true;
    }


    ImGui::EndMenu();
  }

  if(ImGui::BeginMenu("View"))
  {
    if(ImGui::MenuItem(ICON_MS_BOTTOM_PANEL_OPEN "V-Sync", "Ctrl+Shift+V", &vsync))
    {
      doToggleVsync = false;
    }


    ImGui::EndMenu();
  }

  if(doToggleVsync)
  {
    vsync = !vsync;

    m_app->setVsync(vsync);
  }

  if(doOpenFile)
  {
    std::filesystem::path filePath =
        nvgui::windowOpenFileDialog(m_app->getWindowHandle(), "Load supported",
                                    "Supported Files|*.gltf;*.glb;*.cfg|glTF(.gltf, .glb)|*.gltf;*.glb|config file(.cfg)|*.cfg");
    if(!filePath.empty())
    {

      onFileDrop(filePath);
    }
  }

  if(m_scene && doReloadFile)
  {
    std::filesystem::path filePath = m_sceneFilePathDropLast;

    onFileDrop(filePath);
  }

  if(m_scene)
  {
    if(!m_scene->m_loadedFromCache && doSaveCacheFile)
    {

      saveCacheFile();
    }

    if(hasCache && doDeleteCacheFile)
    {
      try
      {
        if(std::filesystem::remove(m_scene->getCacheFilePath()))
        {

          LOGI("Cache file deleted successfully.\n");
        }
      }

      catch(const std::filesystem::filesystem_error& e)
      {
        LOGW("Problem deleting cache file: %s\n", e.what());
      }
    }
  }


  if(doCloseApp)
  {

    m_app->close();
  }
}

}
