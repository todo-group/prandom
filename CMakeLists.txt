cmake_minimum_required(VERSION 3.14)
include(cmake/prefix.cmake)
project(prandom CXX)
include(cmake/postfix.cmake)

find_package(OpenMP)
if(OPENMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif(OPENMP_FOUND)

include_directories(${PROJECT_SOURCE_DIR})

include(standards)
FetchContent_MakeAvailable(${FetchContents})
include_directories(${FetchContent_includes})

add_subdirectory(benchmark)
