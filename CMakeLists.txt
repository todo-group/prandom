cmake_minimum_required(VERSION 3.14)
include(cmake/prefix.cmake)
project(prandom CXX)
include(cmake/postfix.cmake)

find_package(OpenMP)
find_package(MPI)

include_directories(${PROJECT_SOURCE_DIR})

include(standards)
FetchContent_MakeAvailable(${FetchContents})
include_directories(${FetchContent_includes})

add_subdirectory(benchmark)
add_subdirectory(example)
