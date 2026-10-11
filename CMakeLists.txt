cmake_minimum_required(VERSION 3.24)
project(crsim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

include(FetchContent)
FetchContent_Declare(nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(nlohmann_json)

if (MSVC)
  add_compile_options(/W4 /permissive- /utf-8)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Headless simulation core: no rendering, no IO except loading game data.
add_library(crsim_core
  core/src/game_data.cpp
  core/src/battle.cpp
  core/src/bots.cpp
  core/src/rl.cpp
  core/src/batch_env.cpp)
target_include_directories(crsim_core PUBLIC core/include)
target_link_libraries(crsim_core PRIVATE nlohmann_json::nlohmann_json)

# BatchEnv steps matches in parallel with OpenMP; without it, it simply runs serially.
find_package(OpenMP)
if (OpenMP_CXX_FOUND)
  target_link_libraries(crsim_core PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(sim_cli apps/sim_cli/main.cpp)
target_link_libraries(sim_cli PRIVATE crsim_core)

# Python module `crsim` (pybind11). Skipped if Python or pybind11 is not available.
option(CRSIM_PYTHON "Build the crsim Python module" ON)
if (CRSIM_PYTHON)
  find_package(Python 3.10 COMPONENTS Interpreter Development.Module)
  if (Python_FOUND)
    execute_process(
      COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
      OUTPUT_VARIABLE CRSIM_PYBIND11_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET)
    # pybind11 quotes the path when it contains spaces.
    string(REPLACE "\"" "" CRSIM_PYBIND11_DIR "${CRSIM_PYBIND11_DIR}")
    find_package(pybind11 CONFIG HINTS "${CRSIM_PYBIND11_DIR}")
  endif()
  if (pybind11_FOUND)
    set_target_properties(crsim_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
    pybind11_add_module(crsim python/bindings.cpp)
    target_link_libraries(crsim PRIVATE crsim_core)
    # Put the module next to the Python scripts so they can `import crsim` directly.
    set_target_properties(crsim PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python")
  else()
    message(STATUS "crsim: Python/pybind11 not found, skipping the Python module")
  endif()
endif()

enable_testing()
add_executable(crsim_tests tests/test_battle.cpp)
target_link_libraries(crsim_tests PRIVATE crsim_core)
add_test(NAME crsim_tests COMMAND crsim_tests ${CMAKE_SOURCE_DIR}/data)
