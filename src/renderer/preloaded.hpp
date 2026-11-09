//==============================================================================
// src/renderer/preloaded.hpp
// Declares the fully resident scene path.
// Preloading uploads all geometry group payloads once and exposes stable device addresses to traversal and render shaders.
//==============================================================================
#pragma once


#include "scene.hpp"
#include "resources.hpp"


namespace lodclusters {


class ScenePreloaded
{
public:


  struct Config
  {
  };


  static bool canPreload(VkDeviceSize, const Scene* scene);


  bool init(Resources* res, const Scene* scene, const Config& config);


  void deinit();


  const nvvk::BufferTyped<shaderio::Geometry>& getShaderGeometriesBuffer() const { return m_shaderGeometriesBuffer; }


  size_t getGeometrySize() const { return m_geometrySize; }
  size_t getOperationsSize() const { return m_operationsSize; }

private:


  struct Geometry
  {
    nvvk::BufferTyped<shaderio::LodLevel> lodLevels;
    nvvk::BufferTyped<shaderio::Node>     lodNodes;
    nvvk::BufferTyped<shaderio::BBox>     lodNodeBboxes;

    nvvk::Buffer                groupData;
    nvvk::BufferTyped<uint64_t> groupAddresses;
    nvvk::BufferTyped<uint64_t> clusterAddresses;
  };

  Config       m_config;
  Resources*   m_resources = nullptr;
  const Scene* m_scene     = nullptr;

  size_t m_geometrySize       = 0;
  size_t m_operationsSize     = 0;

  std::vector<ScenePreloaded::Geometry> m_geometries;
  std::vector<shaderio::Geometry>       m_shaderGeometries;

  nvvk::BufferTyped<shaderio::Geometry> m_shaderGeometriesBuffer;
};
}
