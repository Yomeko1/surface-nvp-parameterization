cmake_minimum_required(VERSION 3.16)
project(surface_nvp_slim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
set(LIBIGL_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LIBIGL_BUILD_TUTORIALS OFF CACHE BOOL "" FORCE)
set(LIBIGL_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  libigl
  GIT_REPOSITORY https://github.com/libigl/libigl.git
  GIT_TAG v2.6.0
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(libigl)

add_executable(surface_nvp_slim main.cpp)
target_link_libraries(surface_nvp_slim PRIVATE igl::core)
