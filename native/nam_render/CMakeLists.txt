cmake_minimum_required(VERSION 3.18)
project(hybrid_nam_render)

# Fetches NeuralAmpModelerCore source only (not its own tools/CMakeLists.txt,
# which sets an invalid -Wno-error flag under MSVC) and builds the `render`
# CLI tool directly. hybrid/render.py shells out to the resulting exe.
include(FetchContent)
FetchContent_Declare(NAMCore
  GIT_REPOSITORY https://github.com/sdatkinson/NeuralAmpModelerCore.git
  GIT_TAG v0.5.4
)
FetchContent_GetProperties(NAMCore)
if(NOT namcore_POPULATED)
  FetchContent_Populate(NAMCore)
endif()

set(NAM_SRC_DIR "${namcore_SOURCE_DIR}")
set(AUDIO_DSP_TOOLS_DIR "${NAM_SRC_DIR}/Dependencies/AudioDSPTools")

file(GLOB NAM_SOURCES_TOP "${NAM_SRC_DIR}/NAM/*.cpp")
file(GLOB NAM_SOURCES_SUB "${NAM_SRC_DIR}/NAM/*/*.cpp")

add_executable(nam_render
  "${NAM_SRC_DIR}/tools/render.cpp"
  ${NAM_SOURCES_TOP}
  ${NAM_SOURCES_SUB}
  "${AUDIO_DSP_TOOLS_DIR}/dsp/wav.cpp"
)
target_include_directories(nam_render PRIVATE
  "${NAM_SRC_DIR}"
  "${NAM_SRC_DIR}/Dependencies/eigen"
  "${NAM_SRC_DIR}/Dependencies/nlohmann"
  "${AUDIO_DSP_TOOLS_DIR}/dsp"
)
target_compile_features(nam_render PUBLIC cxx_std_20)
target_compile_definitions(nam_render PRIVATE NAM_ENABLE_A2_FAST)
if(WIN32)
  target_compile_definitions(nam_render PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()
