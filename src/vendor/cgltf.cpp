//==============================================================================
// src/vendor/cgltf.cpp
// Owns the single cgltf implementation translation unit for this project.
// Keep third-party implementation macros isolated here so every other file can include cgltf as declarations only.
//==============================================================================
#define _CRT_SECURE_NO_WARNINGS


#define CGLTF_IMPLEMENTATION
#if !defined(NDEBUG)


#define CGLTF_VALIDATE_ENABLE_ASSERTS 1
#endif


#include <cgltf.h>
