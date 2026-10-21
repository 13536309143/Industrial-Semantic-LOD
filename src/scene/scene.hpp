//==============================================================================
// src/scene/scene.hpp
// Declares CPU-side scene storage, cache views, geometry group layouts, instances, cameras, and preprocessing state.
// Scene converts glTF primitives into clustered LOD group payloads that can be either preloaded or streamed to the GPU.
//==============================================================================
#pragma once


#include <vector>
#include <array>
#include <string>
#include <atomic>
#include <chrono>
#include <mutex>
#include <unordered_set>
#include <unordered_map>
#include <functional>
#include <glm/glm.hpp>
#include <nvutils/file_mapping.hpp>
#include <nvutils/timers.hpp>
#include <nvutils/alignment.hpp>
#include "serialization.hpp"
#include "meshlod.h"
#include "shaderio_scene.h"


namespace lodclusters {

enum SemanticLodMode : uint32_t
{
  SEMANTIC_LOD_POINTNEXT = 0,
  SEMANTIC_LOD_POINTCLIP,
  SEMANTIC_LOD_POINTNEXT_STRUCTURE,
  SEMANTIC_LOD_POINTCLIP_STRUCTURE,
  SEMANTIC_LOD_POINTNEXT_POINTCLIP,
  SEMANTIC_LOD_POINTNEXT_POINTCLIP_STRUCTURE,
  SEMANTIC_LOD_POINTNEXT_POINTCLIP_STRUCTURE_METADATA,
};

// Build-time scene settings that affect cache identity and GPU-visible payload layout.
struct SceneConfig
{
  static const uint32_t version = 9;


  uint32_t clusterVertices    = 128;
  uint32_t clusterTriangles   = 128;
  uint32_t clusterGroupSize   = 32;
  uint32_t preferredNodeWidth = 8;


  bool useCompressedData = false;


  uint32_t enabledAttributes = shaderio::CLUSTER_ATTRIBUTE_VERTEX_NORMAL;


  float meshoptFillWeight  = 0.5f;
  float meshoptSplitFactor = 2.0f;


  float lodLevelDecimationFactor = 0.5f;


  float lodErrorMergePrevious = 1.5;
  float lodErrorMergeAdditive = 0.0f;


  float simplifyNormalWeight      = 0.5f;
  float simplifyTangentWeight     = 0.01f;
  float simplifyTangentSignWeight = 0.5f;
  float simplifyTexCoordWeight    = 0;


  uint32_t compressionPosDropBits = 7;
  uint32_t compressionTexDropBits = 7;


  float lodErrorEdgeLimit = 0;


  uint32_t assemblyCullingMinInstances = 8;
  float    assemblyLodPixelThreshold   = 24.0f;

  bool  featureConstraints        = true;
  float featureImportanceWeight   = 4.0f;
  float featureProtectThreshold   = 0.78f;
  float featureCriticalThreshold  = 0.93f;

  uint32_t semanticLodMode = SEMANTIC_LOD_POINTNEXT_POINTCLIP_STRUCTURE;

  uint32_t reservedData[11] = {};


};


struct SceneLoaderConfig
{


  float processingThreadsPct = 0.5;


  bool processingOnly = false;

  bool processingAllowPartial = false;

  int processingMode = 0;


  bool autoSaveCache = true;

  bool autoLoadCache = true;


  bool memoryMappedCache = false;


  size_t forcePreprocessMiB = size_t(2) * 1024;


  std::atomic_uint32_t* progressPct = nullptr;
};


struct SceneGridConfig
{
  static const uint32_t minCopies = 1;
  static const uint32_t maxCopies = 32;


  bool      uniqueGeometriesForCopies = false;
  uint32_t  numCopies                 = 1;
  uint32_t  gridBits                  = 13;
  glm::vec3 refShift                  = {1.0f, 1.0f, 1.0f};
  float     snapAngle                 = 0;
  float     minScale                  = 1.0f;
  float     maxScale                  = 1.0f;
};


// Owns imported scene data and all CPU preprocessing products needed by runtime rendering.
class Scene
{
public:


  enum Result
  {
    SCENE_RESULT_SUCCESS,
    SCENE_RESULT_CACHE_INVALID,
    SCENE_RESULT_NEEDS_PREPROCESS,
    SCENE_RESULT_PREPROCESS_COMPLETED,
    SCENE_RESULT_ERROR,
  };

  Result init(const std::filesystem::path& filePath,
              const SceneConfig&           config,
              const SceneLoaderConfig&     loaderConfig,
              const std::string&           cacheSuffix,
              bool                         skipCache);


  bool   saveCache() const;


  void   deinit();


  void updateSceneGrid(const SceneGridConfig& gridConfig);

  bool isMemoryMappedCache() const { return m_loadedFromCache && m_cacheFileMapping.valid(); }

  const std::filesystem::path& getFilePath() const { return m_filePath; }
  const std::filesystem::path& getCacheFilePath() const { return m_cacheFilePath; }


  struct Range
  {
    uint32_t offset;
    uint32_t count;
  };


  // Compact metadata for a packed group payload stored in the scene cache and upload buffers.
struct GroupInfo
  {
    uint64_t offsetBytes : 42;
    uint64_t sizeBytes : 22;
    uint16_t vertexCount;
    uint16_t triangleCount;
    uint8_t  lodLevel;
    uint8_t  clusterCount;
    uint8_t  attributeBits;
    uint8_t  reserved1 = 0;
    uint64_t vertexDataCount : 21;


    uint64_t uncompressedVertexDataCount : 21;
    uint64_t uncompressedSizeBytes : 22;


    uint32_t getDeviceSize() const { return uint32_t(uncompressedSizeBytes ? uncompressedSizeBytes : sizeBytes); }


    uint32_t estimateVertexDataCount() const
    {
      uint32_t dataCount = vertexCount * 3;
      if(attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_NORMAL)
      {
        dataCount += vertexCount * 1;
      }
      if(attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_0)
      {
        dataCount += vertexCount * 2;
        dataCount += clusterCount;
      }
      if(attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_1)
      {
        dataCount += vertexCount * 2;
        dataCount += clusterCount;
      }
      return dataCount;
    }


    size_t computeSize() const
    {
      size_t threadGroupSize = sizeof(shaderio::Group);
      threadGroupSize        = nvutils::align_up(threadGroupSize, 16) + sizeof(shaderio::Cluster) * clusterCount;
      threadGroupSize        = nvutils::align_up(threadGroupSize, 4) + sizeof(uint32_t) * clusterCount;
      threadGroupSize        = nvutils::align_up(threadGroupSize, 16) + sizeof(shaderio::BBox) * clusterCount;
      threadGroupSize        = threadGroupSize + sizeof(uint8_t) * triangleCount * 3;
      threadGroupSize        = nvutils::align_up(threadGroupSize, 8) + sizeof(float) * vertexDataCount;
      return nvutils::align_up(threadGroupSize, 16);
    }


    size_t computeUncompressedSectionSize() const
    {
      size_t threadGroupSize = sizeof(shaderio::Group);
      threadGroupSize        = nvutils::align_up(threadGroupSize, 16) + sizeof(shaderio::Cluster) * clusterCount;
      threadGroupSize        = nvutils::align_up(threadGroupSize, 4) + sizeof(uint32_t) * clusterCount;
      threadGroupSize        = nvutils::align_up(threadGroupSize, 16) + sizeof(shaderio::BBox) * clusterCount;
      threadGroupSize        = threadGroupSize + sizeof(uint8_t) * triangleCount * 3;

      threadGroupSize        = nvutils::align_up(threadGroupSize, 8);
      return threadGroupSize;
    }
  };


  // Non-owning view that decodes a packed group payload into typed spans without copying.
struct GroupView
  {
    const uint8_t*                     raw     = nullptr;
    const size_t                       rawSize = 0;
    const shaderio::Group*             group   = nullptr;
    std::span<const shaderio::Cluster> clusters;
    std::span<const uint32_t>          clusterGeneratingGroups;
    std::span<const shaderio::BBox>    clusterBboxes;
    std::span<const uint8_t>           indices;
    std::span<const float>             vertices;

    GroupView() {};


    GroupView(std::span<const uint8_t> groupDatas, const GroupInfo& info)

        : rawSize(info.sizeBytes)
    {
      assert(info.offsetBytes + info.sizeBytes <= groupDatas.size());
      raw = &groupDatas[info.offsetBytes];


      size_t startAddress = size_t(raw);

      group = (const shaderio::Group*)raw;
      clusters = std::span((const shaderio::Cluster*)nvutils::align_up(startAddress + sizeof(shaderio::Group), 16), info.clusterCount);
      clusterGeneratingGroups =
          std::span((const uint32_t*)nvutils::align_up(size_t(clusters.data() + info.clusterCount), 4), info.clusterCount);
      clusterBboxes =
          std::span((const shaderio::BBox*)nvutils::align_up(size_t(clusterGeneratingGroups.data() + info.clusterCount), 16),
                    info.clusterCount);

      indices = std::span((const uint8_t*)size_t(clusterBboxes.data() + info.clusterCount), info.triangleCount * 3);

      vertices = std::span((const float*)nvutils::align_up(size_t(indices.data() + info.triangleCount * 3), 8), info.vertexDataCount);
      assert((size_t(vertices.data() + info.vertexDataCount) - startAddress) <= size_t(info.sizeBytes));
    }


    const uint8_t* getClusterIndices(size_t clusterIndex) const
    {

      return (const uint8_t*)(size_t(&clusters[clusterIndex]) + clusters[clusterIndex].indices);
    }


    const glm::vec3* getClusterVertices(size_t clusterIndex) const
    {

      return (const glm::vec3*)(size_t(&clusters[clusterIndex]) + clusters[clusterIndex].vertices);
    }
  };


  struct GroupStorage
  {
    uint8_t*                     raw;
    const size_t                 rawSize = 0;
    shaderio::Group*             group;
    std::span<shaderio::Cluster> clusters;
    std::span<uint32_t>          clusterGeneratingGroups;
    std::span<shaderio::BBox>    clusterBboxes;
    std::span<uint8_t>           indices;
    std::span<float>             vertices;

    GroupStorage() {};


    GroupStorage(void* groupData, const GroupInfo& info)

        : rawSize(info.sizeBytes)
    {
      size_t startAddress = (size_t)groupData;

      raw   = (uint8_t*)groupData;
      group = (shaderio::Group*)startAddress;
      clusters = std::span((shaderio::Cluster*)nvutils::align_up(startAddress + sizeof(shaderio::Group), 16), info.clusterCount);
      clusterGeneratingGroups =
          std::span((uint32_t*)nvutils::align_up(size_t(clusters.data() + info.clusterCount), 4), info.clusterCount);
      clusterBboxes =
          std::span((shaderio::BBox*)nvutils::align_up(size_t(clusterGeneratingGroups.data() + info.clusterCount), 16),
                    info.clusterCount);
      indices = std::span((uint8_t*)size_t(clusterBboxes.data() + info.clusterCount), info.triangleCount * 3);
      vertices = std::span((float*)nvutils::align_up(size_t(indices.data() + info.triangleCount * 3), 8), info.vertexDataCount);
      assert((size_t(vertices.data() + info.vertexDataCount) - startAddress) <= size_t(info.sizeBytes));
    }


    uint32_t getClusterLocalOffset(uint32_t clusterIndex, const void* input, size_t overrideSize = 0) const
    {
      assert(size_t(input) >= size_t(&clusters[clusterIndex]));
      assert(size_t(input) < size_t(raw + (overrideSize ? overrideSize : rawSize)));

      return uint32_t(size_t(input) - size_t(&clusters[clusterIndex]));
    }


    uint32_t* getClusterLocalData(uint32_t clusterIndex, uint32_t localOffset)
    {
      return (uint32_t*)(size_t(&clusters[clusterIndex]) + localOffset);
    }
  };


  static void fillGroupRuntimeData(const GroupInfo& srcGroupInfo,
                                   const GroupView& srcGroupView,
                                   uint32_t         groupID,
                                   uint32_t         groupResidentID,
                                   uint32_t         clusterResidentID,
                                   void*            dst,
                                   size_t           dstSize);


  static void decompressGroup(const GroupInfo& info, const GroupView& groupView, void* dstWriteOnly, size_t dstSize);


  struct GeometryLodInput
  {
    uint64_t inputTriangleCount       = 0;
    uint64_t inputVertexCount         = 0;
    uint64_t inputTriangleIndicesHash = 0;
    uint64_t inputVerticesHash        = 0;
    uint64_t semanticPolicyHash       = 0;
  };


  struct GeometryBase
  {
    uint32_t attributeBits = 0;

    uint32_t clusterMaxVerticesCount{};
    uint32_t clusterMaxTrianglesCount{};

    uint32_t lodLevelsCount{};


    uint32_t hiTriangleCount{};
    uint32_t hiVerticesCount{};
    uint32_t hiClustersCount{};


    uint32_t totalTriangleCount{};
    uint32_t totalVerticesCount{};
    uint32_t totalClustersCount{};

    shaderio::BBox bbox{};

    GeometryLodInput lodInfo;

    uint32_t instanceReferenceCount{};
    float    semanticLodErrorScale = 1.0f;
    uint32_t semanticLodPolicyFlags = 0;
  };


  struct GeometryView : GeometryBase
  {

    std::span<const uint8_t> groupData;


    std::span<const GroupInfo> groupInfos;

    std::span<const shaderio::LodLevel> lodLevels;
    std::span<const shaderio::Node>     lodNodes;
    std::span<const shaderio::BBox>     lodNodeBboxes;


    std::span<const uint32_t> localMaterialIDs;


    inline uint64_t getCachedSize() const
    {
      uint64_t cachedSize = 0;

      cachedSize += (sizeof(GeometryBase) + serialization::ALIGN_MASK) & ~serialization::ALIGN_MASK;

      cachedSize += serialization::getCachedSize(groupData);

      cachedSize += serialization::getCachedSize(groupInfos);

      cachedSize += serialization::getCachedSize(lodLevels);

      cachedSize += serialization::getCachedSize(lodNodes);

      cachedSize += serialization::getCachedSize(lodNodeBboxes);

      cachedSize += serialization::getCachedSize(localMaterialIDs);

      return cachedSize;
    }
  };


  const GeometryView& getActiveGeometry(size_t idx) const { return m_geometryViews[idx % m_originalGeometryCount]; }
  size_t              getActiveGeometryCount() const { return m_activeGeometryCount; }


  uint32_t getGeometryInstanceFactor() const
  {
    return m_gridConfig.uniqueGeometriesForCopies ? 1u : uint32_t(m_instances.size() / m_originalInstanceCount);
  }


  struct Instance
  {
    glm::mat4      matrix;
    shaderio::BBox bbox;
    uint32_t       geometryID = ~0U;
    uint32_t       materialID = ~0U;
    uint32_t       assemblyID = SHADERIO_INVALID_ASSEMBLY;
    bool           twoSided   = false;
    glm::vec4      color{0.8, 0.8, 0.8, 1.0f};
    float          lodErrorScale = 1.0f;
    uint32_t       lodPolicy     = 0;
  };

  struct GltfNodeImportResult
  {
    uint32_t       firstInstance = 0;
    uint32_t       instanceCount = 0;
    shaderio::BBox bbox          = {};
  };

  struct AssemblyTemplate
  {
    uint64_t fingerprint        = 0;
    uint32_t firstAssembly      = SHADERIO_INVALID_ASSEMBLY;
    uint32_t assemblyCount      = 0;
    uint32_t instanceCount      = 0;
  };


  struct Camera
  {
    glm::mat4 worldMatrix{1};
    glm::vec3 eye{0, 0, 0};
    glm::vec3 center{0, 0, 0};
    glm::vec3 up{0, 1, 0};
    float     fovy;
  };


  struct Histograms
  {
    static const uint32_t version = 1;

    std::array<uint32_t, 256 + 1>                         clusterTriangles = {};
    std::array<uint32_t, 256 + 1>                         clusterVertices  = {};
    std::array<uint32_t, SHADERIO_MAX_GROUP_CLUSTERS + 1> groupClusters    = {};
    std::array<uint32_t, SHADERIO_MAX_NODE_CHILDREN + 1>  nodeChildren     = {};
    std::array<uint32_t, SHADERIO_MAX_LOD_LEVELS + 1>     lodLevels        = {};

    uint32_t clusterTrianglesMax = {};
    uint32_t clusterVerticesMax  = {};
    uint32_t groupClustersMax    = {};
    uint32_t nodeChildrenMax     = {};
    uint32_t lodLevelsMax        = {};
  };

  struct ProcessingStatsSnapshot
  {
    static const uint32_t version = 2;

    uint64_t groups                = 0;
    uint64_t clusters              = 0;
    uint64_t vertices              = 0;
    uint64_t groupUniqueVertices   = 0;
    uint64_t groupHeaderBytes      = 0;
    uint64_t triangleIndexBytes    = 0;
    uint64_t vertexPosBytes        = 0;
    uint64_t vertexTexCoordBytes   = 0;
    uint64_t vertexNrmBytes        = 0;
    uint64_t vertexCompressedBytes = 0;
    uint64_t clusterBboxBytes      = 0;
    uint64_t clusterHeaderBytes    = 0;
    uint64_t clusterGenBytes       = 0;
    uint64_t inputFeatureVertices  = 0;
    uint64_t inputFeatureTris      = 0;
    uint64_t boundaryVertices      = 0;
    uint64_t nonManifoldVertices   = 0;
    uint64_t sharpEdgeVertices     = 0;
    uint64_t boundaryComponents    = 0;
    uint64_t sharpRingComponents   = 0;
    uint64_t circularHoleLoops     = 0;
    uint64_t circularHoleVertices  = 0;
    uint64_t functionalBoundaryVertices = 0;
    uint64_t cylindricalVertices   = 0;
    uint64_t thinWallVertices      = 0;
    uint64_t protectedVertices     = 0;
    uint64_t criticalVertices      = 0;
    uint64_t featureImportanceSumPpm = 0;
    uint64_t featureImportanceMaxPpm = 0;

  };


  SceneConfig       m_config;
  SceneLoaderConfig m_loaderConfig;
  SceneGridConfig   m_gridConfig;
  std::string       m_cacheSuffix;

  shaderio::BBox m_bbox;
  shaderio::BBox m_gridBbox;

  std::vector<Instance> m_instances;
  std::vector<shaderio::AssemblyNode> m_assemblyNodes;
  std::vector<AssemblyTemplate>       m_assemblyTemplates;
  std::vector<Camera>   m_cameras;

  bool m_isBig       = false;
  bool m_hasTwoSided = false;

  uint32_t m_maxClusterTriangles       = 0;
  uint32_t m_maxClusterVertices        = 0;
  uint32_t m_maxPerGeometryClusters    = 0;
  uint32_t m_maxPerGeometryTriangles   = 0;
  uint32_t m_maxPerGeometryVertices    = 0;
  uint32_t m_maxLodLevelsCount         = 0;
  uint32_t m_hiPerGeometryClusters     = 0;
  uint32_t m_hiPerGeometryTriangles    = 0;
  uint32_t m_hiPerGeometryVertices     = 0;
  uint64_t m_hiClustersCount           = 0;
  uint64_t m_hiVerticesCount           = 0;
  uint64_t m_hiTrianglesCount          = 0;
  uint64_t m_hiClustersCountInstanced  = 0;
  uint64_t m_hiTrianglesCountInstanced = 0;
  uint64_t m_totalClustersCount        = 0;
  uint64_t m_totalTrianglesCount       = 0;
  uint64_t m_totalVerticesCount        = 0;

  Histograms m_histograms;
  ProcessingStatsSnapshot m_processingStats;

  bool m_loadedFromCache    = false;
  bool m_hasVertexNormals   = false;
  bool m_hasVertexTexCoord0 = false;
  bool m_hasVertexTexCoord1 = false;
  bool m_hasVertexTangents  = false;

  size_t m_originalInstanceCount = 0;
  size_t m_originalGeometryCount = 0;

  size_t m_cacheFileSize = 0;

  struct SemanticLodPolicy
  {
    bool     valid      = false;
    bool     allowCull  = false;
    uint32_t priority   = 3;
    uint32_t flags      = 0;
    uint32_t meshIndex  = ~0u;
    uint32_t nodeIndex  = ~0u;
    float    confidence = 0.0f;
    float    targetRatioNear = 0.70f;
    float    targetRatioMid  = 0.40f;
    float    targetRatioFar  = 0.18f;
    float    screenErrorWeight = 1.0f;
    float    simplifyRatio = 0.50f;
    float    lodErrorScale = 1.0f;
    float    errorMergeScale = 1.0f;
    float    featureWeightScale = 1.0f;
    float    featureProtectThreshold = 0.78f;
    float    featureCriticalThreshold = 0.93f;
    float    featureSoftScale = 1.0f;
    float    featureHardLockRatio = 0.10f;
    // Optional geometry constraints remain independent of semantic confidence.
    float    featureLockFloor = 0.0f;
    float    featureSoftFloor = 0.0f;
    uint32_t schemaVersion = 0;
    float    featureLockBudget = 0.035f;
    float    explicitFeatureSoftScale = 1.0f;
    float    explicitFeatureWeightScale = 1.0f;
    float    explicitProtectThreshold = 0.82f;
    float    explicitCriticalThreshold = 0.98f;
    float    simplifyErrorRelative = 0.0f;
    float    maxAxisShrink = 0.0f;
    bool     preserveComponents = false;
    float    baseSimplifyRatio = 1.0f;
    float    baseErrorRelative = 0.0f;
    float    hierarchyDepthDecay = 0.02f;
    float    hierarchyMinRatio = 0.42f;
    uint32_t partitionSize = 16;
    uint64_t rowHash = 0;
  };

private:


  // Owning storage for one geometry while it is being imported, processed, and serialized.
struct GeometryStorage : GeometryBase
  {

    std::vector<glm::vec3>  vertexPositions;
    std::vector<float>      vertexAttributes;
    std::vector<glm::uvec3> triangles;

    uint32_t attributesWithWeights  = 0u;
    uint32_t attributeNormalOffset  = ~0u;
    uint32_t attributeTex0offset    = ~0u;
    uint32_t attributeTex1offset    = ~0u;
    uint32_t attributeTangentOffset = ~0u;

    SemanticLodPolicy semanticPolicy;
    bool              hasSemanticLodPolicy = false;

    std::vector<uint8_t>   groupData;
    std::vector<GroupInfo> groupInfos;

    std::vector<shaderio::LodLevel> lodLevels;
    std::vector<shaderio::BBox>     lodNodeBboxes;
    std::vector<shaderio::Node>     lodNodes;

    std::vector<uint32_t> localMaterialIDs;
  };

  size_t m_activeGeometryCount = 0;

  std::vector<SemanticLodPolicy> m_semanticMeshPolicies;
  std::unordered_map<uint64_t, SemanticLodPolicy> m_semanticNodePolicies;
  uint64_t m_semanticLodFingerprint = 0;
  std::filesystem::path m_semanticLodPath;

  std::vector<GeometryStorage> m_geometryStorages;
  std::vector<GeometryView>    m_geometryViews;
  std::unordered_map<uint64_t, uint32_t> m_assemblyTemplateMap;


  static bool     loadCached(GeometryView& view, uint64_t dataSize, const void* data);


  static bool     storeCached(const GeometryView& view, uint64_t dataSize, void* data);


  static uint64_t storeCached(const GeometryView& view, FILE* outFile);


  void openCache();


  void closeCache();


  bool checkCache(const GeometryLodInput& info, size_t geometryIndex);


  void loadCachedGeometry(GeometryStorage& geometry, size_t geometryIndex);


  class CacheFileHeader
  {
  public:

    CacheFileHeader()
    {
      memset(this, 0, sizeof(CacheFileHeader));
      header = {};
      config = {};
    }


    bool isValid() const
    {
      Header reference = {};

      return header.magic == reference.magic && header.geoVersion == reference.geoVersion
             && header.geoStructSize == reference.geoStructSize && header.configStructSize == reference.configStructSize
             && header.configVersion == reference.configVersion && header.histogramsVersion == reference.histogramsVersion
             && header.histogramStructSize == reference.histogramStructSize
             && header.processingStatsVersion == reference.processingStatsVersion
             && header.processingStatsStructSize == reference.processingStatsStructSize && header.alignment == reference.alignment;
    }

  private:


    struct Header
    {
      uint64_t magic               = 0x006f65676e73766eULL;
      uint32_t geoVersion          = 12;
      uint32_t geoStructSize       = uint32_t(sizeof(GeometryView));
      uint32_t configVersion       = SceneConfig::version;
      uint32_t configStructSize    = uint32_t(sizeof(SceneConfig));
      uint32_t histogramsVersion   = Histograms::version;
      uint32_t histogramStructSize = uint32_t(sizeof(Histograms));
      uint32_t processingStatsVersion = ProcessingStatsSnapshot::version;
      uint32_t processingStatsStructSize = uint32_t(sizeof(ProcessingStatsSnapshot));
      uint64_t alignment           = serialization::ALIGNMENT;


    };

    Header header;

  public:
    SceneConfig             config;
    Histograms              histograms;
    ProcessingStatsSnapshot processingStats;
    uint32_t                pad[7];
  };

  static_assert(sizeof(CacheFileHeader) % serialization::ALIGNMENT == 0, "CacheFileHeader size unaligned");


  class CacheFileView
  {


#if 0


    struct CacheFile
    {

      CacheHeader header;

      uint8_t geometryViewData[];


      uint64_t geometryOffsets[geometryCount * 2];
      uint64_t geometryCount;
    };
#endif

  public:
    bool isValid() const { return m_dataSize != 0; }


    bool init(uint64_t dataSize, const void* data);

    void deinit() { *(this) = {}; }

    uint64_t getGeometryCount() const { return m_geometryCount; }


    void getSceneConfig(SceneConfig& settings) const;


    void getHistograms(Histograms& histograms) const;

    void getProcessingStats(ProcessingStatsSnapshot& stats) const;


    bool getGeometryView(GeometryView& view, uint64_t geometryIndex) const;

  private:
    template <class T>

    const T* getPointer(uint64_t offset, uint64_t count = 1) const
    {
      assert(offset + sizeof(T) * count <= m_dataSize);
      return reinterpret_cast<const T*>(m_dataBytes + offset);
    }

    uint64_t       m_dataSize      = 0;
    uint64_t       m_tableStart    = 0;
    const uint8_t* m_dataBytes     = nullptr;
    uint64_t       m_geometryCount = 0;
  };


  struct CachePartialEntry
  {
    uint64_t geometryIndex = 0;
    uint64_t offset        = 0;
    uint64_t dataSize      = 0;
  };

  std::filesystem::path m_filePath;
  std::filesystem::path m_cacheFilePath;
  std::filesystem::path m_cachePartialFilePath;


  nvutils::FileReadMapping m_cacheFileMapping;
  CacheFileView            m_cacheFileView;


  FILE*                 m_processingOnlyFile             = nullptr;
  FILE*                 m_processingOnlyPartialFile      = nullptr;
  size_t                m_processingOnlyPartialCompleted = 0;
  uint64_t              m_processingOnlyFileOffset       = 0;
  std::vector<uint64_t> m_processingOnlyGeometryOffsets;


  // Shared progress, timing, threading, and compressed-buffer state for scene preprocessing.
struct ProcessingInfo
  {


    uint32_t numPoolThreadsOriginal = 1;
    uint32_t numPoolThreads         = 1;

    uint32_t numOuterThreads = 1;
    uint32_t numInnerThreads = 1;


    size_t   geometryCount = 0;
    uint64_t triangleCount = 0;

    std::mutex processOnlySaveMutex;


    std::vector<uint32_t> bufferViewUsers;
    std::vector<uint32_t> bufferViewLocks;


    struct Stats
    {
      std::atomic_uint64_t groups                = 0;
      std::atomic_uint64_t clusters              = 0;
      std::atomic_uint64_t vertices              = 0;
      std::atomic_uint64_t groupUniqueVertices   = 0;
      std::atomic_uint64_t groupHeaderBytes      = 0;
      std::atomic_uint64_t triangleIndexBytes    = 0;
      std::atomic_uint64_t vertexPosBytes        = 0;
      std::atomic_uint64_t vertexTexCoordBytes   = 0;
      std::atomic_uint64_t vertexNrmBytes        = 0;
      std::atomic_uint64_t vertexCompressedBytes = 0;
      std::atomic_uint64_t clusterBboxBytes      = 0;
      std::atomic_uint64_t clusterHeaderBytes    = 0;
      std::atomic_uint64_t clusterGenBytes       = 0;
      std::atomic_uint64_t inputFeatureVertices  = 0;
      std::atomic_uint64_t inputFeatureTris      = 0;
      std::atomic_uint64_t boundaryVertices      = 0;
      std::atomic_uint64_t nonManifoldVertices   = 0;
      std::atomic_uint64_t sharpEdgeVertices     = 0;
      std::atomic_uint64_t boundaryComponents    = 0;
      std::atomic_uint64_t sharpRingComponents   = 0;
      std::atomic_uint64_t circularHoleLoops     = 0;
      std::atomic_uint64_t circularHoleVertices  = 0;
      std::atomic_uint64_t functionalBoundaryVertices = 0;
      std::atomic_uint64_t cylindricalVertices   = 0;
      std::atomic_uint64_t thinWallVertices      = 0;
      std::atomic_uint64_t protectedVertices     = 0;
      std::atomic_uint64_t criticalVertices      = 0;
      std::atomic_uint64_t featureImportanceSumPpm = 0;
      std::atomic_uint64_t featureImportanceMaxPpm = 0;
    } stats;


    uint32_t   progressLastPercentage      = 0;
    uint32_t   progressGeometriesCompleted = 0;
    uint64_t   progressTrianglesCompleted  = 0;
    std::mutex progressMutex;
    std::atomic_bool cacheMismatchLogged = false;

    nvutils::PerformanceTimer clock;
    double                    startTime = 0;
    std::chrono::steady_clock::time_point preprocessStartTime;
    clodTimingStats           lodTimingStats;
    std::atomic_uint64_t      quantCacheMicroseconds = 0;
    std::atomic_uint64_t      cacheLoadMicroseconds  = 0;

    static uint64_t timestampMicroseconds()
    {
      return uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(
                          std::chrono::steady_clock::now().time_since_epoch())
                          .count());
    }

    static void addMicroseconds(std::atomic_uint64_t& dst, uint64_t start)
    {
      dst.fetch_add(timestampMicroseconds() - start, std::memory_order_relaxed);
    }


    void init(float pct);


    void setupParallelism(size_t geometryCount_, size_t geometryCompletedCount, int parallelismMode);


    void setupCompressedGltf(size_t bufferViewCount);


    void deinit();


    void     logBegin(uint64_t totalTriangleCount);

    uint32_t logCompletedGeometry(uint64_t triangleCount = 0);


    void     logEnd();
  };


  Result loadGLTF(ProcessingInfo& processingInfo, const std::filesystem::path& filePath);

private:


  void loadGeometryGLTF(ProcessingInfo& processingInfo, uint64_t geometryIndex, size_t meshIndex, const struct cgltf_data* gltf);

  void loadSemanticLodPolicies();
  const SemanticLodPolicy* findSemanticMeshPolicy(uint32_t meshIndex) const;
  const SemanticLodPolicy* findSemanticNodePolicy(uint32_t nodeIndex, uint32_t meshIndex) const;
  uint64_t semanticPolicyHashForMesh(uint32_t meshIndex) const;
  void applySemanticPolicyToConfig(clodConfig& config, const SemanticLodPolicy& policy) const;
  GltfNodeImportResult addInstancesFromNodeGLTF(const std::vector<size_t>& meshToGeometry,
                                                const struct cgltf_data*   data,
                                                const struct cgltf_node*   node,
                                                const glm::mat4 parentObjToWorldTransform = glm::mat4(1),
                                                uint32_t        depth                     = 0);

  void assignAssemblyToRange(uint32_t assemblyID, uint32_t firstInstance, uint32_t instanceCount);


  bool loadCompressedViewsGLTF(ProcessingInfo&                                processingInfo,
                               std::unordered_set<struct cgltf_buffer_view*>& bufferViews,
                               const struct cgltf_data*                       gltf);
  void unloadCompressedViewsGLTF(ProcessingInfo&                                processingInfo,
                                 std::unordered_set<struct cgltf_buffer_view*>& bufferViews,
                                 const struct cgltf_data*                       gltf);


  void processGeometry(ProcessingInfo& processingInfo, size_t geometryIndex, bool isCached);


  void buildGeometryLod(ProcessingInfo& processingInfo, GeometryStorage& geometry);


  void buildHierarchy(ProcessingInfo& processingInfo, GeometryStorage& geometry);


  void computeLodBboxes_recursive(GeometryStorage& geometry, size_t nodeIdx);


  void buildGeometryDedupVertices(ProcessingInfo& processingInfo, GeometryStorage& geometry);


  void computeHistogramMaxs();


  void computeInstanceBBoxes();


  void beginProcessingOnly(size_t geometryCount);


  void saveProcessingOnly(ProcessingInfo& processingInfo, size_t geometryIndex);


  bool endProcessingOnly(ProcessingInfo& processingInfo, bool hadError);


  struct TempContext
  {
    ProcessingInfo&  processingInfo;
    GeometryStorage& geometry;
    Scene&           scene;

    bool      innerThreadingActive   = false;
    bool      levelGroupOffsetValid  = false;
    GroupInfo threadGroupInfo        = {};
    uint32_t  threadGroupSize        = 0;
    uint32_t  threadGroupStorageSize = 0;
    uint32_t  lodLevel               = ~0u;
    size_t    levelGroupOffset       = 0;


    std::mutex           groupMutex;
    std::atomic_uint32_t groupIndexOrdered = 0;
    std::atomic_size_t   groupDataOrdered  = 0;
    std::vector<uint8_t> threadGroupDatas;
  };


  struct TempGroup
  {
    uint32_t                  lodLevel;
    uint32_t                  clusterCount;
    shaderio::TraversalMetric traversalMetric;
  };


  struct TempCluster
  {
    const uint32_t* indices         = nullptr;
    uint32_t        indexCount      = 0;
    uint32_t        generatingGroup = 0;
  };

  uint32_t storeGroup(TempContext*       context,
                      uint32_t           threadIndex,
                      uint32_t           groupIndex,
                      const clodGroup&   group,
                      uint32_t           clusterCount,
                      const clodCluster* clusters);


  void compressGroup(TempContext* context, GroupStorage& groupTempStorage, GroupInfo& groupInfo, uint32_t* vertexCacheLocal);


  static void clodIterationMeshoptimizer(void* iteration_context, void* output_context, int depth, size_t task_count);
  static int  clodGroupMeshoptimizer(void*              output_context,
                                     clodGroup          group,
                                     const clodCluster* clusters,
                                     size_t             cluster_count,
                                     size_t             task_index,
                                     uint32_t           thread_index);
};

}
