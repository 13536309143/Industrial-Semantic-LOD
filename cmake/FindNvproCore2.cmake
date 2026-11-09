# FindNvproCore2.cmake

if(NOT NvproCore2_FOUND)
    set(_Print_info TRUE)
endif()

# Default to the nvpro_core2 revision this project is known to work with.
set(_NVPRO_DEFAULT_GIT_TAG "03297bee1f997208951cc104abb5ecc3e1f987ed")
if(NOT DEFINED NVPRO_GIT_TAG OR NVPRO_GIT_TAG STREQUAL "main")
    set(NVPRO_GIT_TAG "${_NVPRO_DEFAULT_GIT_TAG}" CACHE STRING "Git tag/branch/commit for nvpro_core2" FORCE)
else()
    set(NVPRO_GIT_TAG "${NVPRO_GIT_TAG}" CACHE STRING "Git tag/branch/commit for nvpro_core2")
endif()


# Try to find local installation first
find_path(NvproCore2_ROOT
    NAMES nvpro_core2/cmake/Setup.cmake
    PATHS
    ${CMAKE_BINARY_DIR}/_deps
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/..
    ${CMAKE_SOURCE_DIR}/../..
)

if(NvproCore2_ROOT)
    set(_NvproCore2_DIR "${NvproCore2_ROOT}/nvpro_core2")
    if(EXISTS "${_NvproCore2_DIR}/.git")
        execute_process(
            COMMAND git -C "${_NvproCore2_DIR}" rev-parse HEAD
            OUTPUT_VARIABLE _NVPRO_CURRENT_COMMIT
            RESULT_VARIABLE _NVPRO_REV_PARSE_RESULT
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )

        if(_NVPRO_REV_PARSE_RESULT EQUAL 0 AND NOT _NVPRO_CURRENT_COMMIT STREQUAL NVPRO_GIT_TAG)
            if(_NvproCore2_DIR STREQUAL "${CMAKE_BINARY_DIR}/_deps/nvpro_core2")
                message(STATUS "nvpro_core2 is at ${_NVPRO_CURRENT_COMMIT}; switching to ${NVPRO_GIT_TAG}")
                execute_process(
                    COMMAND git -C "${_NvproCore2_DIR}" fetch --depth 1 origin ${NVPRO_GIT_TAG}
                    RESULT_VARIABLE _NVPRO_FETCH_RESULT
                )
                if(NOT _NVPRO_FETCH_RESULT EQUAL 0)
                    message(FATAL_ERROR "Failed to fetch nvpro_core2 commit ${NVPRO_GIT_TAG}")
                endif()

                execute_process(
                    COMMAND git -C "${_NvproCore2_DIR}" checkout --detach FETCH_HEAD
                    RESULT_VARIABLE _NVPRO_CHECKOUT_RESULT
                )
                if(NOT _NVPRO_CHECKOUT_RESULT EQUAL 0)
                    message(FATAL_ERROR "Failed to checkout nvpro_core2 commit ${NVPRO_GIT_TAG}")
                endif()
            else()
                message(WARNING "Found nvpro_core2 at ${_NvproCore2_DIR}, but it is at ${_NVPRO_CURRENT_COMMIT}; requested ${NVPRO_GIT_TAG}")
            endif()
        endif()
    endif()
    set(NvproCore2_FOUND TRUE)
else()
    # Option to allow downloading if not found
    option(NVPROCORE2_DOWNLOAD "Download nvpro_core2 if not found" ON)

    if(NVPROCORE2_DOWNLOAD)
        # Try to determine nvpro_core location from git remote
        execute_process(
            COMMAND git config --get remote.origin.url
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            OUTPUT_VARIABLE GIT_REPO_URL
            RESULT_VARIABLE GIT_RESULT
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )

        if(NOT GIT_RESULT EQUAL 0)
            message(WARNING "Failed to get git remote URL. Defaulting to GitHub URL.")
            set(GIT_REPO_URL "https://github.com/nvpro-samples/nvpro_core2.git")
            set(GIT_BASE_URL "https://github.com")
            set(NVPRO_GIT_URL ${GIT_REPO_URL})
        else()
            # Check if this is a GitHub repository
            string(FIND "${GIT_REPO_URL}" "github.com" FOUND_INDEX)
            if(FOUND_INDEX GREATER -1)
                # Extract base URL up to github.com
                string(REGEX MATCH ".*github\\.com" GIT_BASE_URL "${GIT_REPO_URL}")
                if(NOT GIT_BASE_URL)
                    message(FATAL_ERROR "Failed to extract GitHub base URL from ${GIT_REPO_URL}")
                endif()

                # Handle SSH vs HTTPS URLs differently
                string(FIND "${GIT_REPO_URL}" "git@" SSH_FOUND_INDEX)
                if(SSH_FOUND_INDEX GREATER -1)
                    # SSH format
                    set(NVPRO_GIT_URL ${GIT_BASE_URL}:nvpro-samples/nvpro_core2.git)
                else()
                    # HTTPS format
                    set(NVPRO_GIT_URL ${GIT_BASE_URL}/nvpro-samples/nvpro_core2.git)
                endif()

                message(STATUS "Using GitHub nvpro_core2 repository")
            else()
                # Internal repository - reconstruct URL preserving the protocol
                string(REGEX MATCH "^[^/]+//[^/]+/" GIT_BASE_URL "${GIT_REPO_URL}")
                if(NOT GIT_BASE_URL)
                    message(FATAL_ERROR "Failed to extract base URL from ${GIT_REPO_URL}")
                endif()

                set(NVPRO_GIT_URL ${GIT_BASE_URL}devtechproviz/nvpro-samples/nvpro_core2.git)
                message(STATUS "Using internal nvpro_core2 repository")
            endif()
        endif()

        if(NOT NVPRO_GIT_URL)
            message(FATAL_ERROR "Failed to construct git URL for nvpro_core2")
        endif()

        message(STATUS "Will clone from: ${NVPRO_GIT_URL} (tag/branch/commit: ${NVPRO_GIT_TAG})")

        # Fetch the requested commit directly so NVPRO_GIT_TAG can be a tag, branch, or SHA.
        file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/_deps/nvpro_core2)
        execute_process(
            COMMAND git init ${CMAKE_BINARY_DIR}/_deps/nvpro_core2
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            RESULT_VARIABLE _NVPRO_INIT_RESULT
        )
        if(NOT _NVPRO_INIT_RESULT EQUAL 0)
            message(FATAL_ERROR "Failed to initialize nvpro_core2 repository")
        endif()

        execute_process(
            COMMAND git -C ${CMAKE_BINARY_DIR}/_deps/nvpro_core2 remote add origin ${NVPRO_GIT_URL}
            RESULT_VARIABLE _NVPRO_REMOTE_RESULT
        )
        if(NOT _NVPRO_REMOTE_RESULT EQUAL 0)
            message(FATAL_ERROR "Failed to set nvpro_core2 remote URL")
        endif()

        execute_process(
            COMMAND git -C ${CMAKE_BINARY_DIR}/_deps/nvpro_core2 fetch --depth 1 origin ${NVPRO_GIT_TAG}
            RESULT_VARIABLE _NVPRO_FETCH_RESULT
        )
        if(NOT _NVPRO_FETCH_RESULT EQUAL 0)
            message(FATAL_ERROR "Failed to fetch nvpro_core2 commit ${NVPRO_GIT_TAG}")
        endif()

        execute_process(
            COMMAND git -C ${CMAKE_BINARY_DIR}/_deps/nvpro_core2 checkout --detach FETCH_HEAD
            RESULT_VARIABLE _NVPRO_CHECKOUT_RESULT
        )
        if(NOT _NVPRO_CHECKOUT_RESULT EQUAL 0)
            message(FATAL_ERROR "Failed to checkout nvpro_core2 commit ${NVPRO_GIT_TAG}")
        endif()

        # Try to find local installation first
        find_path(NvproCore2_ROOT
            NAMES nvpro_core2/cmake/Setup.cmake
            PATHS
            ${CMAKE_BINARY_DIR}/_deps
            ${CMAKE_SOURCE_DIR}
            ${CMAKE_SOURCE_DIR}/..
            ${CMAKE_SOURCE_DIR}/../..
        )

        if(NvproCore2_ROOT)
            set(NvproCore2_FOUND TRUE)
        endif()
    endif()
endif()

if(_Print_info)
    message(STATUS "Found nvpro_core2 at: ${NvproCore2_ROOT}")
endif()

if(NvproCore2_FOUND)
    set(NvproCore2_FOUND TRUE)   

    # Include the setup file which will add all the necessary libraries
    # and create the actual targets (nvpro2::nvvk etc)
    include(${NvproCore2_ROOT}/nvpro_core2/cmake/Setup.cmake)
endif()
