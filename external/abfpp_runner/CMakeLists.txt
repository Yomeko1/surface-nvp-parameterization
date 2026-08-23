cmake_minimum_required(VERSION 3.16)
project(surface_nvp_abfpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
set(OPENABF_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(OPENABF_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(OPENABF_BUILD_DOCS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  openabf
  GIT_REPOSITORY https://github.com/educelab/OpenABF.git
  GIT_TAG v2.1.0
  GIT_SHALLOW TRUE
  EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(openabf)

add_executable(surface_nvp_abfpp main.cpp)
target_link_libraries(surface_nvp_abfpp PRIVATE OpenABF::OpenABF)

if(MSVC)
  target_compile_options(surface_nvp_abfpp PRIVATE /permissive-)
endif()
