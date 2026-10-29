//==============================================================================
// src/app/lodclusters_config.cpp
// Registers command-line and sequencer parameters for scene, renderer, streaming, and capture options.
// The parser writes into edit/current config structs; frame code later reconciles those edits into live resources.
//==============================================================================



#include "lodclusters.hpp"

bool g_verbose = false;


namespace lodclusters {


LodClusters::LodClusters(const Info& info)

    : m_info(info)
{
  nvutils::ProfilerTimeline::CreateInfo createInfo;
  createInfo.name = "graphics";

  m_profilerTimeline = m_info.profilerManager->createTimeline(createInfo);
  m_info.parameterRegistry->add({"scene"}, {".gltf", ".glb", ".cfg"}, &m_sceneFilePathDropNew);
  m_info.parameterRegistry->add({"renderer"}, (int*)&m_tweak.renderer);
  m_info.parameterRegistry->add({"verbose"}, &g_verbose, true);
  m_info.parameterRegistry->add({"resetstats"}, &m_tweak.autoResetTimers);
  m_info.parameterRegistry->add({"supersample"}, &m_tweak.supersample);
  m_info.parameterRegistry->add({"debugui"}, &m_showDebugUI);
  m_info.parameterRegistry->add({"sequencescreenshot", "save screenshot at end of each sequence. 0 disabled (default), 1 full window, 2 rendered viewport"},(int*)&m_sequenceScreenshotMode, true);
  m_info.parameterRegistry->add({"sequencelossless", "write sequence captures directly as lossless PNG"}, &m_sequenceLossless);
  m_info.parameterRegistry->add({"dumpspirv", "dumps compiled spirv into working directory"}, &m_resources.m_dumpSpirv);
  m_info.parameterRegistry->add({"camerastring"}, &m_cameraString);
  m_info.parameterRegistry->add({"cameraspeed"}, &m_cameraSpeed);
  m_info.parameterRegistry->addVector({"sundirection"}, &m_frameConfig.frameConstants.skyParams.sunDirection);
  m_info.parameterRegistry->addVector({"suncolor"}, &m_frameConfig.frameConstants.skyParams.sunColor);
  m_info.parameterRegistry->add({"streaming"}, &m_tweak.useStreaming);
  m_info.parameterRegistry->add({"gridcopies"}, &m_sceneGridConfig.numCopies);
  m_info.parameterRegistry->add({"modelarray"}, &m_sceneGridConfig.numCopies);
  m_info.parameterRegistry->add({"arraycopies"}, &m_sceneGridConfig.numCopies);
  m_info.parameterRegistry->add({"gridconfig"}, &m_sceneGridConfig.gridBits);
  m_info.parameterRegistry->add({"gridunique"}, &m_sceneGridConfig.uniqueGeometriesForCopies);
  m_info.parameterRegistry->add({"clusterconfig"}, (int*)&m_tweak.clusterConfig);
  m_info.parameterRegistry->add({"clustergroupsize"}, &m_sceneConfig.clusterGroupSize);
  m_info.parameterRegistry->add({"simplifyuvweight"}, &m_sceneConfig.simplifyTexCoordWeight);
  m_info.parameterRegistry->add({"simplifynormalweight"}, &m_sceneConfig.simplifyNormalWeight);
  m_info.parameterRegistry->add({"simplifytangentweight"}, &m_sceneConfig.simplifyTangentWeight);
  m_info.parameterRegistry->add({"simplifytangentsignweight"}, &m_sceneConfig.simplifyTangentSignWeight);
  m_info.parameterRegistry->add({"attributes"}, &m_sceneConfig.enabledAttributes);

  m_info.parameterRegistry->add({"loderrormergeprevious"}, &m_sceneConfig.lodErrorMergePrevious);
  m_info.parameterRegistry->add({"loderrormergeadditive"}, &m_sceneConfig.lodErrorMergeAdditive);
  m_info.parameterRegistry->add({"loderroredgelimit"}, &m_sceneConfig.lodErrorEdgeLimit);
  m_info.parameterRegistry->add({"lodnodewidth"}, &m_sceneConfig.preferredNodeWidth);
  m_info.parameterRegistry->add({"loddecimationfactor"}, &m_sceneConfig.lodLevelDecimationFactor);
  m_info.parameterRegistry->add({"assemblymininstances"}, &m_sceneConfig.assemblyCullingMinInstances);
  m_info.parameterRegistry->add({"assemblylodpixels"}, &m_sceneConfig.assemblyLodPixelThreshold);
  m_info.parameterRegistry->add({"meshoptfillweight"}, &m_sceneConfig.meshoptFillWeight);
  m_info.parameterRegistry->add({"featureconstraints"}, &m_sceneConfig.featureConstraints);
  m_info.parameterRegistry->add({"featurequalityguard", "validate automatic protection against ordinary geometry and normal error"}, &m_sceneConfig.featureQualityGuard);
  m_info.parameterRegistry->add({"featureseamprotection", "enable soft seam hints on detected critical rims"}, &m_sceneConfig.featureSeamProtection);
  m_info.parameterRegistry->add({"featureimportanceweight", "bounded source-normal soft metric scale; legacy option name retained"}, &m_sceneConfig.featureImportanceWeight);
  m_info.parameterRegistry->add({"featureanchorbudget", "maximum fraction of welded vertices used as geometric anchors"}, &m_sceneConfig.featureAnchorBudget);
  m_info.parameterRegistry->add({"featureprotectthreshold"}, &m_sceneConfig.featureProtectThreshold);
  m_info.parameterRegistry->add({"featurecriticalthreshold"}, &m_sceneConfig.featureCriticalThreshold);
  m_info.parameterRegistry->add({"semanticlodmode"}, &m_sceneConfig.semanticLodMode);

  m_info.parameterRegistry->add({"preservefinest"}, &m_sceneConfig.reservedData[1]);
  m_info.parameterRegistry->add({"semanticruntimestrength"}, &m_rendererConfig.semanticRuntimeStrength);
  m_info.parameterRegistry->add({"semanticruntimepivot"}, &m_rendererConfig.semanticRuntimePivot);
  m_info.parameterRegistry->add({"parallelgroups"}, &m_rendererConfig.useParallelGroups);
  m_info.parameterRegistry->add({"boundedassemblylod"}, &m_rendererConfig.boundAssemblyLod);
  m_info.parameterRegistry->add({"assemblycompact"}, &m_rendererConfig.useAssemblyCompact);
  m_info.parameterRegistry->add({"assemblyprefilter"}, &m_rendererConfig.useAssemblyPrefilter);
  m_info.parameterRegistry->add({"assemblyearlyexit"}, &m_rendererConfig.useAssemblyEarlyExit);
  m_info.parameterRegistry->add({"lowdetailfastpath"}, &m_rendererConfig.useLowDetailFastPath);
  m_info.parameterRegistry->add({"hizocclusion"}, &m_rendererConfig.useHizOcclusion);
  m_info.parameterRegistry->add({"benchoutput"}, &m_benchOutput);
  m_info.parameterRegistry->add({"benchtrajectory"}, &m_benchTrajectory);
  m_info.parameterRegistry->add({"benchdecisions"}, &m_benchDecisionsFile);
  m_info.parameterRegistry->add({"benchwarmup"}, &m_benchWarmup);
  m_info.parameterRegistry->add({"benchframes"}, &m_benchFrames);
  m_info.parameterRegistry->add({"benchcapture"}, &m_benchCapture);
  m_info.parameterRegistry->add({"benchcoldstreaming"}, &m_benchColdStreaming);
  m_info.parameterRegistry->add({"benchpacems"}, &m_benchPaceMs);
  m_info.parameterRegistry->add({"groupclusterlanes"}, &m_rendererConfig.groupClusterLanes);
  m_info.parameterRegistry->add({"floatdepth"}, &m_resources.preferFloatDepth);
  m_info.parameterRegistry->add({"twopasscostprobe"}, &m_rendererConfig.twoPassCostProbeInterval);
  m_info.parameterRegistry->add({"twopasscosthysteresis"}, &m_rendererConfig.twoPassCostHysteresis);
  m_info.parameterRegistry->add({"twopasscostsmoothing"}, &m_rendererConfig.twoPassCostSmoothing);
  m_info.parameterRegistry->add({"twopasscostwarmup"}, &m_rendererConfig.twoPassCostWarmupFrames);

  m_info.parameterRegistry->add({"loderror"}, &m_frameConfig.lodPixelError);
  m_info.parameterRegistry->add({"shadowray"}, &m_frameConfig.frameConstants.doShadow);
  m_info.parameterRegistry->add({"maxtransfermegabytes"}, (uint32_t*)&m_streamingConfig.maxTransferMegaBytes);
  m_info.parameterRegistry->add({"maxgeomegabytes"}, (uint32_t*)&m_streamingConfig.maxGeometryMegaBytes);
  m_info.parameterRegistry->add({"maxresidentgroups"}, &m_streamingConfig.maxGroups);
  m_info.parameterRegistry->add({"maxframeloadrequests"}, &m_streamingConfig.maxPerFrameLoadRequests);
  m_info.parameterRegistry->add({"maxframeunloadrequests"}, &m_streamingConfig.maxPerFrameUnloadRequests);
  m_info.parameterRegistry->add({"cullederrorscale"}, &m_frameConfig.culledErrorScale);
  m_info.parameterRegistry->add({"culling"}, &m_rendererConfig.useCulling);

  m_info.parameterRegistry->add({"twopassculling"}, &m_rendererConfig.useTwoPassCulling);
  m_info.parameterRegistry->add({"adaptivetwopassculling"}, &m_rendererConfig.useAdaptiveTwoPassCulling);
  m_info.parameterRegistry->add({"twopassminclusters"}, &m_rendererConfig.twoPassMinClusters);
  m_info.parameterRegistry->add({"twopassmatrixdelta"}, &m_rendererConfig.twoPassMatrixDelta);
  m_info.parameterRegistry->add({"adaptivetwopassmode"}, &m_rendererConfig.adaptiveTwoPassMode);
  m_info.parameterRegistry->add({"twopassmotionpixels"}, &m_rendererConfig.twoPassMotionPixels);
  m_info.parameterRegistry->add({"twopassfastpixels"}, &m_rendererConfig.twoPassFastPixels);
  m_info.parameterRegistry->add({"twopassmaxinterval"}, &m_rendererConfig.twoPassMaxInterval);
  m_info.parameterRegistry->add({"forcedinvisculling"}, &m_rendererConfig.useForcedInvisibleCulling);
  m_info.parameterRegistry->add({"separategroups"}, &m_rendererConfig.useSeparateGroups);
  m_info.parameterRegistry->add({"cachingenabledlevels"}, &m_frameConfig.cachingEnabledLevels);
  m_info.parameterRegistry->add({"instancesorting"}, &m_rendererConfig.useSorting);
  m_info.parameterRegistry->add({"renderclusterbits"}, &m_rendererConfig.numRenderClusterBits);
  m_info.parameterRegistry->add({"rendertraversalbits"}, &m_rendererConfig.numTraversalTaskBits);
  m_info.parameterRegistry->add({"visualize"}, &m_frameConfig.visualize);
  m_info.parameterRegistry->add({"renderstats"}, &m_rendererConfig.useRenderStats);
  m_info.parameterRegistry->add({"extmeshshader"}, &m_rendererConfig.useEXTmeshShader);
  m_info.parameterRegistry->add({"forcepreprocessmegabytes"}, (uint32_t*)&m_sceneLoaderConfig.forcePreprocessMiB);
  m_info.parameterRegistry->add({"facetshading"}, &m_tweak.facetShading);
  m_info.parameterRegistry->add({"flipwinding"}, &m_rendererConfig.flipWinding);
  m_info.parameterRegistry->add({"forcetwosided"}, &m_rendererConfig.forceTwoSided);
  m_info.parameterRegistry->add({"autosavecache", "automatically store cache file for loaded scene. default true"},&m_sceneLoaderConfig.autoSaveCache);
  m_info.parameterRegistry->add({"autoloadcache", "automatically load cache file if found. default true"},&m_sceneLoaderConfig.autoLoadCache);
  m_info.parameterRegistry->add({"mappedcache", "work from memory mapped cache file, otherwise load to sysmem. default false"},&m_sceneLoaderConfig.memoryMappedCache);
  m_info.parameterRegistry->add({"processingonly", "directly terminate app once cache file was saved. default false"},&m_sceneLoaderConfig.processingOnly);
  m_info.parameterRegistry->add({"cacheloadonly", "load an existing scene cache and terminate before creating a Vulkan context. default false"},&m_cacheLoadOnly);
  m_info.parameterRegistry->add({"processingpartial", "in processingonly mode also allow partial/resuming processing. default false"},&m_sceneLoaderConfig.processingAllowPartial);
  m_info.parameterRegistry->add({"processingmode", "0 auto, -1 inner (within geometry), +1 outer (over geometries) parallelism. default 0"},&m_sceneLoaderConfig.processingMode);
  m_info.parameterRegistry->add({"processingthreadpct", "float percentage of threads during initial file load and processing into lod clusters, default 0.5 == 50 %"},&m_sceneLoaderConfig.processingThreadsPct);
  m_info.parameterRegistry->add({"compressed"}, &m_sceneConfig.useCompressedData);
  m_info.parameterRegistry->add({"compressedpositionbits"}, &m_sceneConfig.compressionPosDropBits);
  m_info.parameterRegistry->add({"compressedtexcoordbits"}, &m_sceneConfig.compressionTexDropBits);
  m_info.parameterRegistry->add({"cachesuffix", "default is .zippp"}, &m_sceneCacheSuffix);
  {

    static bool dummy;
    m_info.parameterRegistry->add({"twosided", "deprecated - now detecting doubleSided materials - there is a new forcetwosided"},
                                  &dummy);
  }

  m_frameConfig.frameConstants                         = {};
  m_frameConfig.frameConstants.wireThickness           = 2.f;
  m_frameConfig.frameConstants.wireSmoothing           = 1.f;
  m_frameConfig.frameConstants.wireColor               = {118.f / 255.f, 185.f / 255.f, 0.f};
  m_frameConfig.frameConstants.wireStipple             = 0;
  m_frameConfig.frameConstants.wireBackfaceColor       = {0.5f, 0.5f, 0.5f};
  m_frameConfig.frameConstants.wireStippleRepeats      = 5;
  m_frameConfig.frameConstants.wireStippleLength       = 0.5f;
  m_frameConfig.frameConstants.doShadow                = 1;
  m_frameConfig.frameConstants.doWireframe             = 0;
  m_frameConfig.frameConstants.ambientOcclusionRadius  = 0.1f;
  m_frameConfig.frameConstants.ambientOcclusionSamples = 2;
  m_frameConfig.frameConstants.visualize               = VISUALIZE_LOD;
  m_frameConfig.frameConstants.facetShading            = 1;
  m_frameConfig.frameConstants.lightMixer = 0.5f;
  m_frameConfig.frameConstants.skyParams  = {};
  m_frameConfig.frameConstants.time = 0.0f;
  m_frameConfig.frameConstants.deltaTime = 0.0f;
  m_frameConfig.frameConstants.lodTransitionSpeed = 1.0f;
  m_lastAmbientOcclusionSamples = m_frameConfig.frameConstants.ambientOcclusionSamples;
  m_sceneLoaderConfig.progressPct = &m_sceneProgress;
}

}
