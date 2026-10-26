//==============================================================================
// src/scene/clusterlod.cpp
// Bridges scene geometry storage to the clustered LOD builder output callbacks.
// The code packs generated groups into the runtime layout used by cache files, preloading, streaming, and shaders.
//==============================================================================
#include <glm/gtc/constants.hpp>
#include <nvutils/logger.hpp>
#include <nvutils/parallel_work.hpp>
#include <meshoptimizer.h>
#include "scene.hpp"
#include "attribute_encoding.h"


namespace lodclusters {

   template <typename T0, typename T1>


   void padZeroes(std::span<T0>& previous, T1* next)
   {
     {

       size_t padSize = size_t(next) - size_t(previous.data() + previous.size());
       if(padSize)
       {

         memset(previous.data() + previous.size(), 0, padSize);
       }
     }
   }

  uint32_t Scene::storeGroup(TempContext* context,
                             uint32_t           threadIndex,
                             uint32_t           groupIndex,
                             const clodGroup&   group,
                             uint32_t           clusterCount,
                             const clodCluster* clusters)
  {
      const uint64_t quantCacheStart = ProcessingInfo::timestampMicroseconds();
      ProcessingInfo& processing = context->processingInfo;
      GeometryStorage& geometry = context->geometry;
      Scene::GroupInfo groupInfo = {};

     uint32_t level = uint32_t(group.depth);

     uint8_t* groupTempData = &context->threadGroupDatas[context->threadGroupSize * threadIndex];
     Scene::GroupInfo groupTempInfo = context->threadGroupInfo;


     GroupStorage     groupTempStorage(groupTempData, groupTempInfo);


     std::span<uint32_t> vertexCacheEarlyValue((uint32_t*)(groupTempData + context->threadGroupStorageSize), 256);


     std::span<uint32_t> vertexCacheEarlyPos((uint32_t*)vertexCacheEarlyValue.data() + 256, 256);


     std::span<uint32_t> vertexCacheLocal(vertexCacheEarlyPos.data() + 256, m_config.clusterGroupSize * m_config.clusterVertices);
     uint32_t       clusterMaxVerticesCount = 0;
     uint32_t       clusterMaxTrianglesCount = 0;

     shaderio::BBox groupBbox                = {{FLT_MAX, FLT_MAX, FLT_MAX}, {-FLT_MAX, -FLT_MAX, -FLT_MAX}, 0, 0};
     {

       uint32_t triangleOffset = 0;
       uint32_t vertexOffset   = 0;
       uint32_t vertexDataOffset = 0;

       size_t vertexPosBytes      = 0;
       size_t vertexNrmBytes      = 0;
       size_t vertexTexCoordBytes = 0;

       uint32_t attributeStride = uint32_t(geometry.vertexAttributes.size() / geometry.vertexPositions.size());

       for(uint32_t c = 0; c < clusterCount; c++)
       {
         uint32_t* localVertices = &vertexCacheLocal[vertexOffset];
         const clodCluster& tempCluster = clusters[c];
         shaderio::Cluster& groupCluster  = groupTempStorage.clusters[c];

         uint32_t           triangleCount = uint32_t(tempCluster.index_count / 3);
         uint32_t           vertexCount   = 0;
         groupCluster.vertices = vertexDataOffset;
         groupCluster.indices  = triangleOffset * 3;

         memset(vertexCacheEarlyValue.data(), ~0, vertexCacheEarlyValue.size_bytes());

         for(uint32_t i = 0; i < tempCluster.index_count; i++)
         {
           uint32_t vertexIndex = tempCluster.indices[i];
           uint32_t cacheIndex  = ~0;

           uint32_t cacheEarlyValue = vertexCacheEarlyValue[vertexIndex & 0xFF];
           if(cacheEarlyValue == vertexIndex)
           {
             cacheIndex = vertexCacheEarlyPos[vertexIndex & 0xFF];
           }
           else
           {

             for(uint32_t v = 0; v < vertexCount; v++)
             {
               if(localVertices[v] == vertexIndex)
               {
                 cacheIndex = v;
               }
             }
           }

           if(cacheIndex == ~0)
           {
               cacheIndex = vertexCount++;
               localVertices[cacheIndex] = vertexIndex;
               vertexCacheEarlyValue[vertexIndex & 0xFF] = vertexIndex;
               vertexCacheEarlyPos[vertexIndex & 0xFF] = cacheIndex;
           }


           groupTempStorage.indices[i + triangleOffset * 3] = uint8_t(cacheIndex);
         }

         shaderio::BBox bbox = {{FLT_MAX, FLT_MAX, FLT_MAX}, {-FLT_MAX, -FLT_MAX, -FLT_MAX}, FLT_MAX, -FLT_MAX};

         {

           if(m_config.useCompressedData)
           {
             for(uint32_t v = 0; v < vertexCount; v++)
             {
               glm::vec3 pos = geometry.vertexPositions[localVertices[v]];

               bbox.lo = glm::min(bbox.lo, glm::vec3(pos));
               bbox.hi = glm::max(bbox.hi, glm::vec3(pos));
             }
           }
           else
           {
             for(uint32_t v = 0; v < vertexCount; v++)
             {


               glm::vec3 pos = geometry.vertexPositions[localVertices[v]];
               *(glm::vec3*)&groupTempStorage.vertices[vertexDataOffset + v * 3] = pos;

               bbox.lo = glm::min(bbox.lo, glm::vec3(pos));
               bbox.hi = glm::max(bbox.hi, glm::vec3(pos));
             }
           }

           vertexDataOffset += vertexCount * 3;
           vertexPosBytes += sizeof(float) * 3 * vertexCount;
         }

         if(geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_NORMAL)
         {
           if(!m_config.useCompressedData)
           {
             if(geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TANGENT)
             {
               for(uint32_t v = 0; v < vertexCount; v++)
               {

                 glm::vec3 normal = *(const glm::vec3*)(&geometry.vertexAttributes[localVertices[v] * attributeStride + geometry.attributeNormalOffset]);
                 glm::vec4 tangent = *(const glm::vec4*)(&geometry.vertexAttributes[localVertices[v] * attributeStride + geometry.attributeTangentOffset]);


                 uint32_t encoded = shaderio::normal_pack(normal);
                 encoded |= shaderio::tangent_pack(normal, tangent) << ATTRENC_NORMAL_BITS;
                 *(uint32_t*)&groupTempStorage.vertices[vertexDataOffset + v] = encoded;
               }
             }
             else
             {
               for(uint32_t v = 0; v < vertexCount; v++)
               {
                 glm::vec3 tmp =
                     *(const glm::vec3*)(&geometry.vertexAttributes[localVertices[v] * attributeStride + geometry.attributeNormalOffset]);

                 uint32_t encoded= shaderio::normal_pack(tmp);
                 *(uint32_t*)&groupTempStorage.vertices[vertexDataOffset + v] = encoded;
               }
             }
           }
           vertexDataOffset += vertexCount;
           vertexNrmBytes += sizeof(uint32_t) * vertexCount;
         }

         for(uint32_t t = 0; t < 2; t++)
         {
           shaderio::ClusterAttributeBits usedBit =
               t == 0 ? shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_0 : shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_1;
           uint32_t attributeTexOffset = t == 0 ? geometry.attributeTex0offset : geometry.attributeTex1offset;
           if(geometry.attributeBits & usedBit)
           {

             vertexDataOffset = (vertexDataOffset + 1) & ~1;
             if(!m_config.useCompressedData)
             {
               for(uint32_t v = 0; v < vertexCount; v++)
               {
                 glm::vec2 tmp =*(const glm::vec2*)(&geometry.vertexAttributes[localVertices[v] * attributeStride + attributeTexOffset]);

                 *(glm::vec2*)&groupTempStorage.vertices[vertexDataOffset + v * 2] = tmp;
               }
             }
             vertexDataOffset += vertexCount * 2;
             vertexTexCoordBytes += sizeof(float) * 2 * vertexCount;
           }
         }

         for(uint32_t t = 0; t < triangleCount; t++)
         {
           glm::vec3 trianglePositions[3];
           for(uint32_t v = 0; v < 3; v++)
           {

             trianglePositions[v] =geometry.vertexPositions[localVertices[groupTempStorage.indices[(triangleOffset + t) * 3 + v]]];
           }
           for(uint32_t e = 0; e < 3; e++)
           {

             float distance    = glm::distance(trianglePositions[e], trianglePositions[(e + 1) % 3]);

             bbox.shortestEdge = std::min(bbox.shortestEdge, distance);

             bbox.longestEdge  = std::max(bbox.longestEdge, distance);
           }
         }


         groupBbox.lo = glm::min(groupBbox.lo, bbox.lo);

         groupBbox.hi = glm::max(groupBbox.hi, bbox.hi);

         groupTempStorage.clusterBboxes[c]           = bbox;

         groupTempStorage.clusterGeneratingGroups[c] = uint32_t(tempCluster.refined);

         groupCluster.triangleCountMinusOne = uint8_t(triangleCount - 1);

         groupCluster.vertexCountMinusOne   = uint8_t(vertexCount - 1);

         groupCluster.lodLevel              = uint8_t(level);

         groupCluster.groupChildIndex       = uint8_t(c);

         groupCluster.attributeBits         = uint8_t(geometry.attributeBits);

         groupCluster.localMaterialID       = uint8_t(0);
         groupCluster.reserved              = 0;


         clusterMaxTrianglesCount = std::max(clusterMaxTrianglesCount, triangleCount);

         clusterMaxVerticesCount  = std::max(clusterMaxVerticesCount, vertexCount);

         ((std::atomic_uint32_t&)m_histograms.clusterTriangles[triangleCount])++;
         ((std::atomic_uint32_t&)m_histograms.clusterVertices[vertexCount])++;
         vertexOffset += vertexCount;
         triangleOffset += triangleCount;
       }

       groupInfo.offsetBytes                 = 0;
       groupInfo.reserved1                   = 0;

       groupInfo.clusterCount                = uint8_t(clusterCount);

       groupInfo.triangleCount               = uint16_t(triangleOffset);

       groupInfo.vertexCount                 = uint16_t(vertexOffset);

       groupInfo.lodLevel                    = uint8_t(level);

       groupInfo.attributeBits               = uint8_t(geometry.attributeBits);
       groupInfo.vertexDataCount             = vertexDataOffset;
       groupInfo.uncompressedVertexDataCount = 0;
       groupInfo.uncompressedSizeBytes       = 0;

       groupInfo.sizeBytes                   = groupInfo.computeSize();

       {
         processing.stats.groups++;
         processing.stats.clusters += clusterCount;
         processing.stats.vertices += vertexOffset;
         processing.stats.groupHeaderBytes += sizeof(shaderio::Group);
         processing.stats.clusterHeaderBytes += sizeof(shaderio::Cluster) * clusterCount;
         processing.stats.clusterBboxBytes += sizeof(shaderio::BBox) * clusterCount;
         processing.stats.clusterGenBytes += sizeof(uint32_t) * clusterCount;
         processing.stats.triangleIndexBytes += sizeof(uint8_t) * triangleOffset * 3;
         processing.stats.vertexPosBytes += vertexPosBytes;
         processing.stats.vertexNrmBytes += vertexNrmBytes;
         processing.stats.vertexTexCoordBytes += vertexTexCoordBytes;
         ((std::atomic_uint32_t&)m_histograms.groupClusters[clusterCount])++;
       }

       if(m_config.useCompressedData)
       {
         compressGroup(context, groupTempStorage, groupInfo, vertexCacheLocal.data());
       }
     }


     bool useOrderedLock = groupIndex != ~0 && context->innerThreadingActive;
     if(useOrderedLock)
     {

       while(true)
       {
         if(context->groupIndexOrdered.load() == groupIndex)
         {


           groupInfo.offsetBytes = context->groupDataOrdered.fetch_add(groupInfo.sizeBytes);


           context->groupIndexOrdered.store(groupIndex + 1);
           break;
         }
         else
         {

           std::this_thread::yield();
         }
       }
     }
     {


       std::lock_guard lock(context->groupMutex);


       geometry.bbox.lo = glm::min(groupBbox.lo, geometry.bbox.lo);

       geometry.bbox.hi = glm::max(groupBbox.hi, geometry.bbox.hi);

       if(context->lodLevel != uint32_t(group.depth))
       {

         context->lodLevel                  = uint32_t(group.depth);
         const shaderio::LodLevel* previous = group.depth ? &geometry.lodLevels[group.depth - 1] : nullptr;
         shaderio::LodLevel initLevel{};

         initLevel.clusterOffset = previous ? previous->clusterOffset + previous->clusterCount : 0;
         initLevel.groupOffset   = previous ? previous->groupOffset + previous->groupCount : 0;
         initLevel.minBoundingSphereRadius = FLT_MAX;
         initLevel.minMaxQuadricError      = FLT_MAX;

         geometry.lodLevels.push_back(initLevel);
       }

       geometry.lodLevels[level].clusterCount += groupInfo.clusterCount;
       geometry.lodLevels[level].groupCount++;


       geometry.lodLevels[level].minBoundingSphereRadius =std::min(geometry.lodLevels[level].minBoundingSphereRadius, group.simplified.radius);

       geometry.lodLevels[level].minMaxQuadricError =std::min(geometry.lodLevels[level].minMaxQuadricError, group.simplified.error);


       geometry.clusterMaxTrianglesCount = std::max(clusterMaxTrianglesCount, geometry.clusterMaxTrianglesCount);

       geometry.clusterMaxVerticesCount  = std::max(clusterMaxVerticesCount, geometry.clusterMaxVerticesCount);

       if(level == 0)
       {
         geometry.hiClustersCount += groupInfo.clusterCount;
         geometry.hiTriangleCount += groupInfo.triangleCount;
         geometry.hiVerticesCount += groupInfo.vertexCount;
       }

       geometry.totalClustersCount += groupInfo.clusterCount;
       geometry.totalTriangleCount += groupInfo.triangleCount;
       geometry.totalVerticesCount += groupInfo.vertexCount;

       if(useOrderedLock)
       {
         if(geometry.groupData.size()<groupInfo.offsetBytes + groupInfo.sizeBytes)
         {

           geometry.groupData.resize(groupInfo.offsetBytes + groupInfo.sizeBytes);
         }
       }
       else
       {


         groupInfo.offsetBytes = geometry.groupData.size();

         geometry.groupData.resize(groupInfo.offsetBytes + groupInfo.sizeBytes);
         if(groupIndex == ~0)
         {
           groupIndex = uint32_t(geometry.groupInfos.size());

           geometry.groupInfos.resize(groupIndex + 1);
         }
       }
       geometry.groupInfos[groupIndex] = groupInfo;
       {


         GroupStorage groupStorage(&geometry.groupData[groupInfo.offsetBytes], groupInfo);

         size_t startAddress = size_t(groupStorage.group);
         groupStorage.group->residentID        = 0;
         groupStorage.group->clusterResidentID = 0;

         groupStorage.group->lodLevel                             = level;
         groupStorage.group->clusterCount                         = groupInfo.clusterCount;
         groupStorage.group->traversalMetric.boundingSphereX      = group.simplified.center[0];
         groupStorage.group->traversalMetric.boundingSphereY      = group.simplified.center[1];
         groupStorage.group->traversalMetric.boundingSphereZ      = group.simplified.center[2];
         groupStorage.group->traversalMetric.boundingSphereRadius = group.simplified.radius;
         groupStorage.group->traversalMetric.maxQuadricError      = group.simplified.error;

         memcpy(groupStorage.clusters.data(), groupTempStorage.clusters.data(), groupStorage.clusters.size_bytes());

         for(uint32_t c = 0; c < clusterCount; c++)
         {
           shaderio::Cluster& groupCluster = groupStorage.clusters[c];
           if(groupInfo.uncompressedSizeBytes)
           {
             groupCluster.vertices = groupStorage.getClusterLocalOffset(c, groupStorage.vertices.data() + groupCluster.vertices,groupInfo.uncompressedSizeBytes);
             groupCluster.indices = groupStorage.getClusterLocalOffset(c, groupStorage.vertices.data() + groupCluster.indices);
           }
           else
           {
             groupCluster.vertices = groupStorage.getClusterLocalOffset(c, groupStorage.vertices.data() + groupCluster.vertices);
             groupCluster.indices = groupStorage.getClusterLocalOffset(c, groupStorage.indices.data() + groupCluster.indices);
           }
         }

         memcpy(groupStorage.clusterGeneratingGroups.data(), groupTempStorage.clusterGeneratingGroups.data(),groupStorage.clusterGeneratingGroups.size_bytes());
         padZeroes(groupStorage.clusterGeneratingGroups, groupStorage.clusterBboxes.data());

         memcpy(groupStorage.clusterBboxes.data(), groupTempStorage.clusterBboxes.data(), groupStorage.clusterBboxes.size_bytes());
         memcpy(groupStorage.indices.data(), groupTempStorage.indices.data(), groupStorage.indices.size_bytes());
         padZeroes(groupStorage.indices, groupStorage.vertices.data());
         memcpy(groupStorage.vertices.data(), groupTempStorage.vertices.data(), groupStorage.vertices.size_bytes());
         padZeroes(groupStorage.vertices, (uint32_t*)(groupStorage.raw + groupInfo.sizeBytes));
       }
     }
     ProcessingInfo::addMicroseconds(processing.quantCacheMicroseconds, quantCacheStart);
     return groupIndex;
  }


   void Scene::clodIterationMeshoptimizer(void* intermediate_context, void* output_context, int depth, size_t task_count)
   {
     TempContext*     context  = reinterpret_cast<TempContext*>(output_context);
     GeometryStorage& geometry = context->geometry;


     context->levelGroupOffset      = geometry.groupInfos.size();
     context->levelGroupOffsetValid = true;

     geometry.groupInfos.resize(context->levelGroupOffset + task_count);


     nvutils::parallel_batches_pooled<1>(task_count,[&](uint64_t idx, uint32_t threadInnerIdx)
         {

           clodBuild_iterationTask(intermediate_context, output_context, idx, threadInnerIdx);
         },
         context->processingInfo.numInnerThreads);
         context->levelGroupOffsetValid = false;
   }


   int Scene::clodGroupMeshoptimizer(void* output_context, clodGroup group, const clodCluster* clusters, size_t cluster_count, size_t task_index, uint32_t thread_index)
   {
     TempContext*     context  = reinterpret_cast<TempContext*>(output_context);
     GeometryStorage& geometry = context->geometry;

     uint32_t groupIndex =context->innerThreadingActive && context->levelGroupOffsetValid ? uint32_t(context->levelGroupOffset + task_index) : ~0u;

     return context->scene.storeGroup(context, thread_index, groupIndex, group, uint32_t(cluster_count), clusters);
   }


   void Scene::buildGeometryLod(ProcessingInfo& processingInfo, GeometryStorage& geometry)
   {


     clodConfig clodInfo = clodDefaultConfig(m_config.clusterTriangles);

     clodInfo.cluster_fill_weight = m_config.meshoptFillWeight;
     clodInfo.cluster_split_factor = m_config.meshoptSplitFactor;
     clodInfo.max_vertices = m_config.clusterVertices;
     clodInfo.partition_size = m_config.clusterGroupSize;
     clodInfo.partition_spatial = true;
     clodInfo.partition_sort = true;
     clodInfo.optimize_clusters = true;

     while((clodInfo.partition_size + clodInfo.partition_size / 3) > m_config.clusterGroupSize)
     {
       clodInfo.partition_size--;
     }

     clodInfo.simplify_error_merge_previous = m_config.lodErrorMergePrevious;
     clodInfo.simplify_error_merge_additive = m_config.lodErrorMergeAdditive;
     clodInfo.simplify_error_edge_limit = m_config.lodErrorEdgeLimit;
     clodInfo.feature_constraints = m_config.featureConstraints;
     clodInfo.feature_quality_guard = m_config.featureQualityGuard;
     clodInfo.feature_seam_protection = m_config.featureSeamProtection;
     clodInfo.feature_attribute_weight = m_config.featureImportanceWeight;
     clodInfo.feature_hard_lock_ratio = std::max(0.0f, std::min(0.35f, m_config.featureAnchorBudget));
     clodInfo.feature_protect_threshold = m_config.featureProtectThreshold;
     clodInfo.feature_critical_threshold = m_config.featureCriticalThreshold;

     if(geometry.hasSemanticLodPolicy)
     {
       applySemanticPolicyToConfig(clodInfo, geometry.semanticPolicy);
       while((clodInfo.partition_size + clodInfo.partition_size / 3) > m_config.clusterGroupSize)
       {
         clodInfo.partition_size--;
       }
       geometry.semanticLodErrorScale = geometry.semanticPolicy.lodErrorScale;
       geometry.semanticLodPolicyFlags = geometry.semanticPolicy.flags;
     }


     clodMesh inputMesh                = {};
     inputMesh.vertex_positions        = reinterpret_cast<const float*>(geometry.vertexPositions.data());

     inputMesh.vertex_count            = geometry.vertexPositions.size();
     inputMesh.vertex_positions_stride = sizeof(glm::vec3);
     inputMesh.index_count             = geometry.triangles.size() * 3;
     inputMesh.indices                 = reinterpret_cast<const uint32_t*>(geometry.triangles.data());
     clodFeatureMetrics featureMetrics = {};
     inputMesh.feature_metrics         = &featureMetrics;
     inputMesh.timing_stats            = &processingInfo.lodTimingStats;
     float attributeWeights[9] = {};

     if(geometry.attributesWithWeights)
     {

       if(m_config.simplifyNormalWeight > 0 && (geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_NORMAL))
       {
         attributeWeights[geometry.attributeNormalOffset + 0] = m_config.simplifyNormalWeight;
         attributeWeights[geometry.attributeNormalOffset + 1] = m_config.simplifyNormalWeight;
         attributeWeights[geometry.attributeNormalOffset + 2] = m_config.simplifyNormalWeight;
       }

       if(m_config.simplifyTexCoordWeight > 0 && (geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_0))
       {
         attributeWeights[geometry.attributeTex0offset + 0] = m_config.simplifyTexCoordWeight;
         attributeWeights[geometry.attributeTex0offset + 1] = m_config.simplifyTexCoordWeight;
       }
       if(m_config.simplifyTexCoordWeight > 0 && (geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_1))
       {
         attributeWeights[geometry.attributeTex1offset + 0] = m_config.simplifyTexCoordWeight;
         attributeWeights[geometry.attributeTex1offset + 1] = m_config.simplifyTexCoordWeight;
       }

       if(m_config.simplifyTangentWeight > 0 && m_config.simplifyTangentSignWeight > 0 && (geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TANGENT))
       {
         attributeWeights[geometry.attributeTangentOffset + 0] = m_config.simplifyTangentWeight;
         attributeWeights[geometry.attributeTangentOffset + 1] = m_config.simplifyTangentWeight;
         attributeWeights[geometry.attributeTangentOffset + 2] = m_config.simplifyTangentWeight;
         attributeWeights[geometry.attributeTangentOffset + 3] = m_config.simplifyTangentSignWeight;
       }
       inputMesh.attribute_count          = geometry.attributesWithWeights;

       inputMesh.vertex_attributes        = geometry.vertexAttributes.data();
       inputMesh.vertex_attributes_stride = sizeof(float) * inputMesh.attribute_count;
       inputMesh.attribute_weights        = attributeWeights;
     }
     TempContext context = {processingInfo, geometry, *this};

     GroupInfo worstGroup       = {};

     worstGroup.clusterCount    = uint8_t(m_config.clusterGroupSize);

     worstGroup.vertexCount     = uint16_t(m_config.clusterGroupSize * m_config.clusterVertices);

     worstGroup.triangleCount   = uint16_t(m_config.clusterGroupSize * m_config.clusterTriangles);
     worstGroup.attributeBits   = geometry.attributeBits;

     worstGroup.vertexDataCount = worstGroup.estimateVertexDataCount();

     worstGroup.sizeBytes       = worstGroup.computeSize();

     context.innerThreadingActive   = processingInfo.numInnerThreads > 1;
     context.threadGroupInfo        = worstGroup;
     context.threadGroupStorageSize = uint32_t(worstGroup.computeSize());
     context.threadGroupSize        = nvutils::align_up(context.threadGroupStorageSize, 4) + sizeof(uint32_t) * 256 * 2 + sizeof(uint32_t) * m_config.clusterGroupSize * m_config.clusterVertices;

     context.threadGroupDatas.resize(context.threadGroupSize * processingInfo.numInnerThreads);

     size_t reservedClusters  = (geometry.triangles.size() + m_config.clusterTriangles - 1) / m_config.clusterTriangles;
     size_t reservedGroups    = (reservedClusters + m_config.clusterGroupSize - 1) / m_config.clusterGroupSize;

     size_t reservedTriangles = geometry.triangles.size();
     reservedClusters  = size_t(double(reservedClusters) * 2.0);
     reservedGroups    = size_t(double(reservedGroups) * 3.0);
     reservedTriangles = size_t(double(reservedTriangles) * 2.0);
     size_t reservedData = 0;
     reservedData += sizeof(shaderio::Group) * reservedGroups;
     reservedData += sizeof(shaderio::Cluster) * reservedClusters;
     reservedData += sizeof(shaderio::BBox) * reservedClusters;
     reservedData += sizeof(uint32_t) * reservedClusters;
     reservedData += sizeof(uint8_t) * reservedTriangles;
     reservedData += sizeof(glm::vec3) * reservedClusters * m_config.clusterVertices;

     geometry.groupData.reserve(reservedData);

     geometry.groupInfos.reserve(reservedGroups);

     geometry.lodLevels.reserve(32);


     clodBuild(clodInfo, inputMesh, &context, clodGroupMeshoptimizer, nullptr);

     processingInfo.stats.inputFeatureVertices += featureMetrics.input_feature_vertices;
     processingInfo.stats.inputFeatureTris += featureMetrics.input_feature_tris;
     processingInfo.stats.boundaryVertices += featureMetrics.boundary_vertices;
     processingInfo.stats.nonManifoldVertices += featureMetrics.non_manifold_vertices;
     processingInfo.stats.sharpEdgeVertices += featureMetrics.sharp_edge_vertices;
     processingInfo.stats.boundaryComponents += featureMetrics.boundary_components;
     processingInfo.stats.sharpRingComponents += featureMetrics.sharp_ring_components;
     processingInfo.stats.circularHoleLoops += featureMetrics.circular_hole_loops;
     processingInfo.stats.circularHoleVertices += featureMetrics.circular_hole_vertices;
     processingInfo.stats.functionalBoundaryVertices += featureMetrics.functional_boundary_vertices;
     processingInfo.stats.cylindricalVertices += featureMetrics.cylindrical_vertices;
     processingInfo.stats.thinWallVertices += featureMetrics.thin_wall_vertices;
     processingInfo.stats.protectedVertices += featureMetrics.protected_vertices;
     processingInfo.stats.criticalVertices += featureMetrics.critical_vertices;
     processingInfo.stats.featureImportanceSumPpm += featureMetrics.feature_importance_sum_ppm;
     uint64_t previousMax = processingInfo.stats.featureImportanceMaxPpm.load();
     while(previousMax < featureMetrics.feature_importance_max_ppm
           && !processingInfo.stats.featureImportanceMaxPpm.compare_exchange_weak(previousMax, featureMetrics.feature_importance_max_ppm))
     {
     }

     geometry.triangles        = {};
     geometry.vertexPositions  = {};
     geometry.vertexAttributes = {};
     geometry.lodLevelsCount = uint32_t(geometry.lodLevels.size());


     geometry.groupInfos.shrink_to_fit();

     geometry.groupData.shrink_to_fit();

     geometry.lodLevels.shrink_to_fit();


     buildHierarchy(processingInfo, geometry);

     geometry.lodNodeBboxes.resize(geometry.lodNodes.size());

     computeLodBboxes_recursive(geometry, 0);
     ((std::atomic_uint32_t&)m_histograms.lodLevels[geometry.lodLevelsCount])++;
   }


   void Scene::buildHierarchy(ProcessingInfo& processingInfo, GeometryStorage& geometry)
   {


     uint32_t lodLevelCount = geometry.lodLevelsCount;


     std::vector<Range> lodNodeRanges(lodLevelCount);
     {
       uint32_t nodeOffset = 1 + lodLevelCount;
       for(uint32_t lodLevel = 0; lodLevel < lodLevelCount; lodLevel++)
       {
         const shaderio::LodLevel& lodLevelInfo = geometry.lodLevels[lodLevel];
         uint32_t nodeCount = lodLevelInfo.groupCount;
         uint32_t iterationCount = nodeCount;

         while(iterationCount > 1)
         {
           iterationCount = (iterationCount + m_config.preferredNodeWidth - 1) / m_config.preferredNodeWidth;
           nodeCount += iterationCount;
         }
         nodeCount--;
         lodNodeRanges[lodLevel].offset = nodeOffset;
         lodNodeRanges[lodLevel].count  = nodeCount;
         nodeOffset += nodeCount;
       }


       geometry.lodNodes.resize(nodeOffset);
     }

     nvutils::parallel_batches_pooled<1>(
lodLevelCount,[&](uint64_t idx, uint32_t threadInnerIdx)
     {

       uint32_t                  lodLevel     = uint32_t(idx);
       const shaderio::LodLevel& lodLevelInfo = geometry.lodLevels[lodLevel];
       const Range&              lodNodeRange = lodNodeRanges[lodLevel];
       uint32_t nodeCount      = lodLevelInfo.groupCount;
       uint32_t nodeOffset     = lodNodeRange.offset;
       uint32_t lastNodeOffset = nodeOffset;

       for(uint32_t g = 0; g < nodeCount; g++)
       {
         uint32_t         groupID   = g + lodLevelInfo.groupOffset;
         const GroupInfo& groupInfo = geometry.groupInfos[groupID];


         GroupView        groupView(geometry.groupData, groupInfo);

         shaderio::Node& node = nodeCount == 1 ? geometry.lodNodes[1 + lodLevel] : geometry.lodNodes[nodeOffset++];
         node                                      = {};
         node.groupRange.isGroup                   = 1;
         node.groupRange.groupIndex                = groupID;
         node.groupRange.groupClusterCountMinusOne = groupInfo.clusterCount - 1;
         node.traversalMetric                      = groupView.group->traversalMetric;
       }
       if(nodeCount == 1)
       {
         nodeOffset++;
       }

       uint32_t depth = 0;
       uint32_t iterationCount = nodeCount;
       std::vector<uint32_t> partitionedIndices;
       std::vector<shaderio::Node> oldNodes;
       while(iterationCount > 1)
       {
         uint32_t lastNodeCount = iterationCount;
         shaderio::Node* lastNodes = &geometry.lodNodes[lastNodeOffset];


         partitionedIndices.resize(lastNodeCount);
         meshopt_spatialClusterPoints(partitionedIndices.data(), &lastNodes->traversalMetric.boundingSphereX,lastNodeCount, sizeof(shaderio::Node), m_config.preferredNodeWidth);
         {


           oldNodes.clear();
           oldNodes.insert(oldNodes.end(), lastNodes, lastNodes + lastNodeCount);
           for(uint32_t n = 0; n < lastNodeCount; n++)
           {
             lastNodes[n] = oldNodes[partitionedIndices[n]];
           }
         }

         iterationCount = (lastNodeCount + m_config.preferredNodeWidth - 1) / m_config.preferredNodeWidth;

         shaderio::Node* newNodes = iterationCount == 1 ? &geometry.lodNodes[1 + lodLevel] : &geometry.lodNodes[nodeOffset];

         for(uint32_t n = 0; n < iterationCount; n++)
         {
           shaderio::Node& node          = newNodes[n];
           shaderio::Node* childrenNodes = &lastNodes[n * m_config.preferredNodeWidth];

           uint32_t childCount = std::min((n + 1) * m_config.preferredNodeWidth, lastNodeCount) - n * m_config.preferredNodeWidth;
           node                                 = {};
           node.nodeRange.isGroup               = 0;
           node.nodeRange.childCountMinusOne    = childCount - 1;
           node.nodeRange.childOffset           = lastNodeOffset + n * m_config.preferredNodeWidth;
           node.traversalMetric.maxQuadricError = 0;

           for(uint32_t c = 0; c < childCount; c++)
           {

             node.traversalMetric.maxQuadricError =std::max(node.traversalMetric.maxQuadricError, childrenNodes[c].traversalMetric.maxQuadricError);
           }

           meshopt_Bounds merged = meshopt_computeSphereBounds(&childrenNodes[0].traversalMetric.boundingSphereX, childCount, sizeof(shaderio::Node),&childrenNodes[0].traversalMetric.boundingSphereRadius, sizeof(shaderio::Node));
           node.traversalMetric.boundingSphereX      = merged.center[0];
           node.traversalMetric.boundingSphereY      = merged.center[1];
           node.traversalMetric.boundingSphereZ      = merged.center[2];
           node.traversalMetric.boundingSphereRadius = merged.radius;
         }
         lastNodeOffset = nodeOffset;
         nodeOffset += iterationCount;
         depth++;
       }
       nodeOffset--;

       assert(lodNodeRange.offset + lodNodeRange.count == nodeOffset);
     },processingInfo.numInnerThreads);


     {

       meshopt_Bounds merged =meshopt_computeSphereBounds(&geometry.lodNodes[1].traversalMetric.boundingSphereX, lodLevelCount, sizeof(shaderio::Node),&geometry.lodNodes[1].traversalMetric.boundingSphereRadius, sizeof(shaderio::Node));
       shaderio::Node& node          = geometry.lodNodes[0];
       shaderio::Node* childrenNodes = &geometry.lodNodes[1];
       node                                      = {};
       node.nodeRange.isGroup                    = 0;
       node.nodeRange.childCountMinusOne         = lodLevelCount - 1;
       node.nodeRange.childOffset                = 1;
       node.traversalMetric.boundingSphereX      = merged.center[0];
       node.traversalMetric.boundingSphereY      = merged.center[1];
       node.traversalMetric.boundingSphereZ      = merged.center[2];
       node.traversalMetric.boundingSphereRadius = merged.radius;
       node.traversalMetric.maxQuadricError= 0;
       for(uint32_t c = 0; c < lodLevelCount; c++)
       {

         node.traversalMetric.maxQuadricError =std::max(node.traversalMetric.maxQuadricError, childrenNodes[c].traversalMetric.maxQuadricError);
       }
     }
   }
}
