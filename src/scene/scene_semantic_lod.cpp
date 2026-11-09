//==============================================================================
// src/scene/scene_semantic_lod.cpp
// Loads optional semantic LOD policy tables and merges them into mesh preprocessing settings.
// Policies tune simplification priorities per mesh or node while contributing to cache identity through stable hashes.
//==============================================================================
#include "scene.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include <nvutils/file_operations.hpp>
#include <nvutils/logger.hpp>

namespace lodclusters {
namespace {

uint64_t hashCombine(uint64_t seed, uint64_t value)
{
  value ^= value >> 33;
  value *= 0xff51afd7ed558ccdULL;
  value ^= value >> 33;
  value *= 0xc4ceb9fe1a85ec53ULL;
  value ^= value >> 33;
  return seed ^ (value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
}

uint64_t hashString(uint64_t seed, const std::string& text)
{
  for(unsigned char ch : text)
  {
    seed ^= uint64_t(ch);
    seed *= 1099511628211ULL;
  }
  return seed;
}

uint64_t semanticNodeKey(uint32_t nodeIndex, uint32_t meshIndex)
{
  return (uint64_t(nodeIndex) << 32) | uint64_t(meshIndex);
}

std::vector<std::string> splitCsvLine(const std::string& line)
{
  std::vector<std::string> fields;
  std::string current;
  bool quoted = false;

  for(size_t i = 0; i < line.size(); ++i)
  {
    const char ch = line[i];
    if(ch == '"')
    {
      if(quoted && i + 1 < line.size() && line[i + 1] == '"')
      {
        current.push_back('"');
        ++i;
      }
      else
      {
        quoted = !quoted;
      }
    }
    else if(ch == ',' && !quoted)
    {
      fields.push_back(current);
      current.clear();
    }
    else
    {
      current.push_back(ch);
    }
  }

  fields.push_back(current);
  return fields;
}

std::string stripBom(std::string text)
{
  if(text.size() >= 3 && uint8_t(text[0]) == 0xEF && uint8_t(text[1]) == 0xBB && uint8_t(text[2]) == 0xBF)
  {
    text.erase(0, 3);
  }
  return text;
}

bool parseBool(const std::string& value)
{
  std::string lower;
  lower.reserve(value.size());
  for(char ch : value)
    lower.push_back(char(std::tolower(uint8_t(ch))));
  return lower == "true" || lower == "1" || lower == "yes";
}

uint32_t parseUint(const std::string& value, uint32_t fallback = ~0u)
{
  try
  {
    if(value.empty())
      return fallback;
    return uint32_t(std::stoul(value));
  }
  catch(...)
  {
    return fallback;
  }
}

float parseFloat(const std::string& value, float fallback = 0.0f)
{
  try
  {
    if(value.empty())
      return fallback;
    return std::stof(value);
  }
  catch(...)
  {
    return fallback;
  }
}

uint32_t parsePriority(const std::string& value)
{
  std::string lower;
  lower.reserve(value.size());
  for(char ch : value)
    lower.push_back(char(std::tolower(uint8_t(ch))));

  if(lower.find("p5_preserve") != std::string::npos || lower.find("critical_preserve") != std::string::npos)
    return 10;
  if(lower.find("p4_conservative") != std::string::npos)
    return 8;
  if(lower.find("p3_standard") != std::string::npos)
    return 5;
  if(lower.find("p2_aggressive") != std::string::npos)
    return 3;
  if(lower.find("p1_micro") != std::string::npos)
    return 1;

  uint32_t priority = 0;
  bool foundDigit = false;
  for(char ch : value)
  {
    if(ch >= '0' && ch <= '9')
    {
      priority = priority * 10u + uint32_t(ch - '0');
      foundDigit = true;
    }
    else if(foundDigit)
    {
      break;
    }
  }
  if(!foundDigit)
    return 5;
  return std::min<uint32_t>(std::max<uint32_t>(priority, 1), 10);
}

float clampf(float value, float lo, float hi)
{
  return std::min(std::max(value, lo), hi);
}

void derivePolicy(Scene::SemanticLodPolicy& policy)
{
  static const float simplifyByPriority[] = {0.50f, 0.34f, 0.40f, 0.44f, 0.48f, 0.52f, 0.58f, 0.62f, 0.66f, 0.72f, 0.82f};
  static const float mergeByPriority[]    = {1.00f, 0.72f, 0.80f, 0.88f, 0.95f, 1.00f, 1.08f, 1.15f, 1.22f, 1.32f, 1.45f};
  static const float featureByPriority[]  = {1.00f, 0.45f, 0.55f, 0.65f, 0.80f, 1.00f, 1.12f, 1.25f, 1.35f, 1.50f, 1.70f};
  static const float protectByPriority[]  = {0.78f, 0.97f, 0.94f, 0.90f, 0.86f, 0.78f, 0.74f, 0.70f, 0.66f, 0.62f, 0.58f};
  static const float criticalByPriority[] = {0.93f, 0.995f, 0.985f, 0.97f, 0.95f, 0.93f, 0.90f, 0.88f, 0.85f, 0.82f, 0.78f};
  static const float softByPriority[]     = {1.00f, 0.35f, 0.45f, 0.60f, 0.75f, 1.00f, 1.15f, 1.30f, 1.45f, 1.65f, 1.80f};
  static const float lockByPriority[]     = {1.00f, 0.010f, 0.018f, 0.035f, 0.060f, 0.090f, 0.120f, 0.160f, 0.190f, 0.240f, 0.300f};
  static const uint32_t partitionByPriority[] = {16u, 24u, 24u, 22u, 20u, 16u, 14u, 12u, 11u, 10u, 8u};

  const uint32_t priority = std::min<uint32_t>(std::max<uint32_t>(policy.priority, 1), 10);
  // P1 permits reversible runtime omission, never permanent source deletion.
  // Derive this template capability for older CSVs that predate the runtime path.
  policy.allowCull = priority == 1;
  policy.flags = SEMANTIC_LOD_VALID_BIT | ((priority & SEMANTIC_LOD_PRIORITY_MASK) << SEMANTIC_LOD_PRIORITY_SHIFT);
  if(policy.allowCull)
    policy.flags |= SEMANTIC_LOD_ALLOW_CULL_BIT;
  if(priority <= 3)
    policy.flags |= SEMANTIC_LOD_AGGRESSIVE_BIT;
  if(priority >= 8)
    policy.flags |= SEMANTIC_LOD_PRESERVE_BIT;
  if(policy.confidence > 0.0f && policy.confidence < 0.40f)
    policy.flags |= SEMANTIC_LOD_LOW_CONF_BIT;

  const float stableRatio = simplifyByPriority[priority];
  const float nearRatio = clampf(policy.targetRatioNear, 0.24f, 0.96f);
  const float midRatio  = clampf(policy.targetRatioMid, 0.06f, nearRatio);
  const float farRatio  = clampf(policy.targetRatioFar, 0.02f, midRatio);

  policy.simplifyRatio = clampf(nearRatio * 0.85f + stableRatio * 0.15f, 0.24f, 0.96f);
  policy.errorMergeScale = mergeByPriority[priority];
  policy.featureWeightScale = featureByPriority[priority];
  policy.featureProtectThreshold = protectByPriority[priority];
  policy.featureCriticalThreshold = criticalByPriority[priority];
  policy.featureSoftScale = softByPriority[priority];
  policy.featureHardLockRatio = lockByPriority[priority];
  policy.hierarchyDepthDecay = clampf(std::max((nearRatio - midRatio) * 0.50f, (midRatio - farRatio) * 0.35f), 0.0f, 0.16f);
  policy.hierarchyMinRatio = farRatio;
  policy.partitionSize = partitionByPriority[priority];

  if(policy.confidence > 0.0f && policy.confidence < 0.40f)
  {
    policy.featureHardLockRatio *= 0.65f;
    policy.featureSoftScale *= 0.85f;
  }

  policy.featureHardLockRatio = std::max(policy.featureHardLockRatio, clampf(policy.featureLockFloor, 0.0f, 0.35f));
  policy.featureSoftScale = std::max(policy.featureSoftScale, clampf(policy.featureSoftFloor, 0.0f, 2.0f));

  if(policy.schemaVersion >= 4)
  {
    // V4 profiles already combine function and geometry. Do not silently add the
    // legacy per-class locks or reduce geometric protection with semantic confidence.
    policy.simplifyRatio = nearRatio;
    policy.featureHardLockRatio = clampf(policy.featureLockBudget, 0.0f, 0.35f);
    policy.featureSoftScale = clampf(policy.explicitFeatureSoftScale, 0.30f, 2.0f);
    policy.featureWeightScale = clampf(policy.explicitFeatureWeightScale, 0.25f, 2.0f);
    policy.featureProtectThreshold = clampf(policy.explicitProtectThreshold, 0.0f, 1.0f);
    policy.featureCriticalThreshold = clampf(policy.explicitCriticalThreshold, policy.featureProtectThreshold, 1.0f);
    policy.errorMergeScale = 1.0f;
  }

  const float weight = std::max(policy.screenErrorWeight, 0.25f);
  policy.lodErrorScale = clampf(1.0f / weight, policy.schemaVersion >= 4 ? 0.10f : 0.45f, 2.40f);
}

void mergeMeshPolicy(Scene::SemanticLodPolicy& dst, const Scene::SemanticLodPolicy& src)
{
  if(!dst.valid)
  {
    dst = src;
    dst.nodeIndex = ~0u;
    derivePolicy(dst);
    return;
  }

  dst.priority = std::max(dst.priority, src.priority);
  dst.allowCull = dst.allowCull && src.allowCull;
  dst.confidence = std::max(dst.confidence, src.confidence);
  dst.targetRatioNear = std::max(dst.targetRatioNear, src.targetRatioNear);
  dst.targetRatioMid = std::max(dst.targetRatioMid, src.targetRatioMid);
  dst.targetRatioFar = std::max(dst.targetRatioFar, src.targetRatioFar);
  dst.screenErrorWeight = std::max(dst.screenErrorWeight, src.screenErrorWeight);
  dst.featureLockFloor = std::max(dst.featureLockFloor, src.featureLockFloor);
  dst.featureSoftFloor = std::max(dst.featureSoftFloor, src.featureSoftFloor);
  dst.schemaVersion = std::max(dst.schemaVersion, src.schemaVersion);
  dst.featureLockBudget = std::max(dst.featureLockBudget, src.featureLockBudget);
  dst.explicitFeatureSoftScale = std::max(dst.explicitFeatureSoftScale, src.explicitFeatureSoftScale);
  dst.explicitFeatureWeightScale = std::max(dst.explicitFeatureWeightScale, src.explicitFeatureWeightScale);
  dst.explicitProtectThreshold = std::min(dst.explicitProtectThreshold, src.explicitProtectThreshold);
  dst.explicitCriticalThreshold = std::min(dst.explicitCriticalThreshold, src.explicitCriticalThreshold);
  auto stricterLimit = [](float a, float b) { return a <= 0.0f ? b : b <= 0.0f ? a : std::min(a, b); };
  dst.simplifyErrorRelative = stricterLimit(dst.simplifyErrorRelative, src.simplifyErrorRelative);
  dst.maxAxisShrink = stricterLimit(dst.maxAxisShrink, src.maxAxisShrink);
  dst.preserveComponents = dst.preserveComponents || src.preserveComponents;
  dst.baseSimplifyRatio = std::max(dst.baseSimplifyRatio, src.baseSimplifyRatio);
  dst.baseErrorRelative = stricterLimit(dst.baseErrorRelative, src.baseErrorRelative);
  dst.rowHash = hashCombine(dst.rowHash, src.rowHash);
  derivePolicy(dst);
}

std::filesystem::path normalizeCandidate(const std::filesystem::path& path)
{
  std::error_code ec;
  return std::filesystem::weakly_canonical(path, ec);
}

void pushCandidate(std::vector<std::filesystem::path>& candidates, std::unordered_set<std::string>& seen, const std::filesystem::path& path)
{
  const std::filesystem::path normalized = normalizeCandidate(path);
  const std::string key = nvutils::utf8FromPath(normalized);
  if(seen.insert(key).second)
    candidates.push_back(normalized);
}

void pushCsvNameCandidates(std::vector<std::filesystem::path>& candidates,
                           std::unordered_set<std::string>&     seen,
                           const std::filesystem::path&         sceneDir,
                           const std::filesystem::path&         cwd,
                           const std::string&                   csvName)
{
  pushCandidate(candidates, seen, sceneDir / csvName);
  pushCandidate(candidates, seen, sceneDir / "lod_analysis_outputs" / csvName);
  pushCandidate(candidates, seen, sceneDir / "lod_analysis_outputs" / "lod_ablation" / csvName);
  pushCandidate(candidates, seen, sceneDir.parent_path() / "lod_analysis_outputs" / csvName);
  pushCandidate(candidates, seen, sceneDir.parent_path() / "lod_analysis_outputs" / "lod_ablation" / csvName);
  pushCandidate(candidates, seen, cwd / "lod_analysis_outputs" / csvName);
  pushCandidate(candidates, seen, cwd / "lod_analysis_outputs" / "lod_ablation" / csvName);
  pushCandidate(candidates, seen, cwd.parent_path() / "lod_analysis_outputs" / csvName);
  pushCandidate(candidates, seen, cwd.parent_path() / "lod_analysis_outputs" / "lod_ablation" / csvName);
  pushCandidate(candidates, seen, cwd.parent_path().parent_path() / "lod_analysis_outputs" / csvName);
  pushCandidate(candidates, seen, cwd.parent_path().parent_path() / "lod_analysis_outputs" / "lod_ablation" / csvName);
}

const char* semanticModeSuffix(uint32_t mode)
{
  switch(mode)
  {
    case SEMANTIC_LOD_POINTNEXT:
      return "_lod_constraints_1_pointnext.csv";
    case SEMANTIC_LOD_POINTCLIP:
      return "_lod_constraints_2_pointclip.csv";
    case SEMANTIC_LOD_POINTNEXT_STRUCTURE:
      return "_lod_constraints_3_pointnext_structure.csv";
    case SEMANTIC_LOD_POINTCLIP_STRUCTURE:
      return "_lod_constraints_4_pointclip_structure.csv";
    case SEMANTIC_LOD_POINTNEXT_POINTCLIP:
      return "_lod_constraints_5_pointnext_pointclip.csv";
    case SEMANTIC_LOD_POINTNEXT_POINTCLIP_STRUCTURE:
      return "_lod_constraints_6_pointnext_pointclip_structure.csv";
    case SEMANTIC_LOD_POINTNEXT_POINTCLIP_STRUCTURE_METADATA:
      return "_lod_constraints_7_pointnext_pointclip_structure_metadata.csv";
    default:
      return "_lod_constraints_6_pointnext_pointclip_structure.csv";
  }
}

const char* semanticModeLabel(uint32_t mode)
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
      return "PointNeXt + open semantics + geometry + assembly/CAD context";
    default:
      return "PointNeXt + PointCLIP + structure";
  }
}

}

void Scene::loadSemanticLodPolicies()
{
  m_semanticMeshPolicies.clear();
  m_semanticNodePolicies.clear();
  m_semanticLodFingerprint = 0;
  m_semanticLodPath.clear();

  const std::string stem = m_filePath.stem().string();
  const std::string selectedCsvName = stem + semanticModeSuffix(m_config.semanticLodMode);
  const std::string fusedCsvName    = stem + "_lod_constraints_fused.csv";
  const std::string csvName         = stem + "_lod_constraints.csv";
  std::vector<std::filesystem::path> candidates;
  std::unordered_set<std::string> seen;
  const std::filesystem::path sceneDir = m_filePath.parent_path();
  const std::filesystem::path cwd = std::filesystem::current_path();

  pushCsvNameCandidates(candidates, seen, sceneDir, cwd, selectedCsvName);
  if(m_config.semanticLodMode != SEMANTIC_LOD_POINTNEXT_POINTCLIP_STRUCTURE_METADATA)
  {
    pushCsvNameCandidates(candidates, seen, sceneDir, cwd, fusedCsvName);
    pushCsvNameCandidates(candidates, seen, sceneDir, cwd, csvName);
  }

  std::filesystem::path csvPath;
  for(const std::filesystem::path& candidate : candidates)
  {
    std::error_code ec;
    if(std::filesystem::exists(candidate, ec))
    {
      csvPath = candidate;
      break;
    }
  }

  if(csvPath.empty())
  {
    LOGI("Semantic LOD: no constraints CSV found for %s using mode %s\n",
         m_filePath.filename().string().c_str(), semanticModeLabel(m_config.semanticLodMode));
    return;
  }

  std::ifstream file(csvPath);
  if(!file)
  {
    LOGW("Semantic LOD: failed to open %s\n", nvutils::utf8FromPath(csvPath).c_str());
    return;
  }

  std::string headerLine;
  if(!std::getline(file, headerLine))
    return;

  std::vector<std::string> headers = splitCsvLine(stripBom(headerLine));
  std::unordered_map<std::string, size_t> headerIndex;
  for(size_t i = 0; i < headers.size(); ++i)
    headerIndex[headers[i]] = i;

  auto column = [&](const std::vector<std::string>& row, const char* name) -> std::string {
    auto it = headerIndex.find(name);
    if(it == headerIndex.end() || it->second >= row.size())
      return {};
    return row[it->second];
  };

  uint64_t fingerprint = 0x73656d616e746963ULL;
  std::unordered_map<uint32_t, SemanticLodPolicy> meshPolicies;
  size_t rows = 0;
  std::string line;
  while(std::getline(file, line))
  {
    if(line.empty())
      continue;

    std::vector<std::string> row = splitCsvLine(line);
    SemanticLodPolicy policy;
    policy.valid = true;
    policy.meshIndex = parseUint(column(row, "mesh_index"));
    policy.nodeIndex = parseUint(column(row, "node_index"));
    if(policy.meshIndex == ~0u || policy.nodeIndex == ~0u)
      continue;

    policy.priority = parsePriority(column(row, "lod_priority"));
    policy.allowCull = parseBool(column(row, "allow_cull"));
    policy.confidence = parseFloat(column(row, "confidence"), 0.0f);
    policy.targetRatioNear = parseFloat(column(row, "target_ratio_near"), policy.targetRatioNear);
    policy.targetRatioMid = parseFloat(column(row, "target_ratio_mid"), policy.targetRatioMid);
    policy.targetRatioFar = parseFloat(column(row, "target_ratio_far"), policy.targetRatioFar);
    policy.screenErrorWeight = parseFloat(column(row, "screen_error_weight"), policy.screenErrorWeight);
    policy.featureLockFloor = parseFloat(column(row, "feature_lock_floor"), 0.0f);
    policy.featureSoftFloor = parseFloat(column(row, "feature_soft_floor"), 0.0f);
    policy.schemaVersion = parseUint(column(row, "policy_schema"), 0);
    policy.featureLockBudget = parseFloat(column(row, "feature_lock_budget"), policy.featureLockFloor);
    policy.explicitFeatureSoftScale = parseFloat(column(row, "feature_soft_scale"), 1.0f);
    policy.explicitFeatureWeightScale = parseFloat(column(row, "feature_weight_scale"), 1.0f);
    policy.explicitProtectThreshold = parseFloat(column(row, "feature_protect_threshold"), 0.82f);
    policy.explicitCriticalThreshold = parseFloat(column(row, "feature_critical_threshold"), 0.98f);
    policy.simplifyErrorRelative = clampf(parseFloat(column(row, "simplify_error_relative")), 0.0f, 0.1f);
    policy.maxAxisShrink = clampf(parseFloat(column(row, "max_axis_shrink")), 0.0f, 0.25f);
    policy.preserveComponents = parseBool(column(row, "preserve_components"));
    policy.baseSimplifyRatio = clampf(parseFloat(column(row, "base_simplify_ratio"), 1.0f), 0.1f, 1.0f);
    policy.baseErrorRelative = clampf(parseFloat(column(row, "base_error_relative")), 0.0f, 0.01f);
    policy.rowHash = hashString(hashCombine(hashCombine(policy.meshIndex, policy.nodeIndex), policy.priority), line);
    derivePolicy(policy);

    fingerprint = hashCombine(fingerprint, policy.rowHash);
    m_semanticNodePolicies[semanticNodeKey(policy.nodeIndex, policy.meshIndex)] = policy;
    mergeMeshPolicy(meshPolicies[policy.meshIndex], policy);
    ++rows;
  }

  if(meshPolicies.empty())
  {
    LOGW("Semantic LOD: %s had no usable rows\n", nvutils::utf8FromPath(csvPath).c_str());
    return;
  }

  uint32_t maxMeshIndex = 0;
  for(const auto& item : meshPolicies)
    maxMeshIndex = std::max(maxMeshIndex, item.first);

  m_semanticMeshPolicies.assign(size_t(maxMeshIndex) + 1, {});
  for(const auto& item : meshPolicies)
    m_semanticMeshPolicies[item.first] = item.second;

  m_semanticLodFingerprint = hashCombine(fingerprint, rows);
  m_semanticLodPath = csvPath;

  LOGI("Semantic LOD: loaded %zu rows, %zu mesh policies using mode %s from %s\n", rows, meshPolicies.size(),
       semanticModeLabel(m_config.semanticLodMode), nvutils::utf8FromPath(csvPath).c_str());
}

const Scene::SemanticLodPolicy* Scene::findSemanticMeshPolicy(uint32_t meshIndex) const
{
  if(meshIndex < m_semanticMeshPolicies.size() && m_semanticMeshPolicies[meshIndex].valid)
    return &m_semanticMeshPolicies[meshIndex];
  return nullptr;
}

const Scene::SemanticLodPolicy* Scene::findSemanticNodePolicy(uint32_t nodeIndex, uint32_t meshIndex) const
{
  auto it = m_semanticNodePolicies.find(semanticNodeKey(nodeIndex, meshIndex));
  if(it != m_semanticNodePolicies.end() && it->second.valid)
    return &it->second;
  return findSemanticMeshPolicy(meshIndex);
}

uint64_t Scene::semanticPolicyHashForMesh(uint32_t meshIndex) const
{
  const SemanticLodPolicy* policy = findSemanticMeshPolicy(meshIndex);
  if(!policy)
    return 0;
  uint64_t hash=hashCombine(m_semanticLodFingerprint, policy->rowHash);
  // A lossless finest level must never silently reuse a pre-simplified cache.
  if(m_config.reservedData[1]) hash=hashCombine(hash,0x5052455345525645ull);
  return hash;
}

void Scene::applySemanticPolicyToConfig(clodConfig& config, const SemanticLodPolicy& policy) const
{
  if(!policy.valid)
    return;

  const float budgetContrast = 1.0f + 0.35f * clampf(policy.confidence, 0.0f, 1.0f);
  config.simplify_ratio = clampf(powf(clampf(policy.simplifyRatio, 0.02f, 0.96f), budgetContrast), 0.12f, 0.96f);
  config.simplify_error_merge_previous *= policy.errorMergeScale;
  config.semantic_priority = int(std::min<uint32_t>(std::max<uint32_t>(policy.priority, 1), 10));
  config.semantic_confidence = policy.confidence;
  config.feature_soft_scale = clampf(policy.featureSoftScale, 0.30f, 2.00f);
  config.feature_hard_lock_ratio = clampf(policy.featureHardLockRatio, 0.0f, 0.35f);
  config.hierarchy_depth_decay = clampf(policy.hierarchyDepthDecay, 0.0f, 0.16f);
  config.hierarchy_min_ratio = clampf(policy.hierarchyMinRatio, 0.02f, 0.96f);
  config.partition_size = std::max<size_t>(8, std::min<size_t>(24, policy.partitionSize));
  if(policy.schemaVersion >= 4)
  {
    config.simplify_error_relative = policy.simplifyErrorRelative;
    config.max_axis_shrink = policy.maxAxisShrink;
    config.preserve_components = policy.preserveComponents;
    config.base_simplify_ratio = policy.baseSimplifyRatio;
    config.base_error_relative = policy.baseErrorRelative;
    if(m_config.reservedData[1])
    {
      config.base_simplify_ratio=1.0f;
      config.base_error_relative=0.0f;
    }
    // Sloppy fallback and error truncation invalidate a measured quality budget.
    config.simplify_fallback_sloppy = false;
    config.simplify_error_edge_limit = 0.0f;
  }

  if(config.feature_constraints)
  {
    config.feature_attribute_weight *= policy.featureWeightScale;
    config.feature_protect_threshold = policy.featureProtectThreshold;
    config.feature_critical_threshold = policy.featureCriticalThreshold;
  }

  if(policy.priority <= 2)
  {
    config.simplify_regularize = false;
  }
}

}
