cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
get_filename_component(PROJECT_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
project(${PROJECT_NAME} LANGUAGES C CXX)
option(NVPRO2_ENABLE_nvgl OFF)
option(NVPRO2_ENABLE_nvgpu_monitor OFF)
option(NVPRO2_ENABLE_nvslang OFF)
option(NVPRO2_ENABLE_nvvkgltf OFF)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(NvproCore2 REQUIRED)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_CXX_STANDARD 20)
file(GLOB_RECURSE SOURCE_FILES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h"
)
file(GLOB_RECURSE SHADER_FILES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.glsl"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.inc"
)
file(GLOB VK_RADIX_SORT_FILES thirdparty/vulkan_radix_sort/src/vk_radix_sort.cc)
add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${SHADER_FILES} ${VK_RADIX_SORT_FILES})
target_include_directories(${PROJECT_NAME} PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/src/app"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/core"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/meshlod"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/scene"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/streaming"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/vendor"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/interface"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/common"
  "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty"
  "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/vulkan_radix_sort/include"
)
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})
target_link_libraries(${PROJECT_NAME} PRIVATE
  nvpro2::nvapp
  nvpro2::nvgui
  nvpro2::nvutils
  nvpro2::nvvk
  nvpro2::nvvkglsl
  meshoptimizer
  cgltf
)
# CPU-only diagnostic probe for offline semantic-policy geometry experiments.
add_library(semantic_geometry_probe SHARED tools/pointclipv2/native/semantic_geometry_probe.cpp src/meshlod/meshlod.cpp)
target_include_directories(semantic_geometry_probe PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/meshlod")
target_link_libraries(semantic_geometry_probe PRIVATE meshoptimizer)
set_target_properties(semantic_geometry_probe PROPERTIES FOLDER "Tools")
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/src" PREFIX "Source Files" FILES ${SOURCE_FILES})
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/shaders" PREFIX "Shader Files" FILES ${SHADER_FILES})
add_project_definitions(${PROJECT_NAME})
copy_to_runtime_and_install( ${PROJECT_NAME}
    FILES ${DLSS_DLLS}
    NVSHADERS_FILES "${NVSHADERS_DIR}/nvshaders/slang_types.h"
                    "${NVSHADERS_DIR}/nvshaders/sky_io.h.slang"
                    "${NVSHADERS_DIR}/nvshaders/sky_functions.h.slang"
                    "${NVSHADERS_DIR}/nvshaders/constants.h.slang"
    LOCAL_DIRS "${CMAKE_CURRENT_LIST_DIR}/shaders"
    AUTO
)

