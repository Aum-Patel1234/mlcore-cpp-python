set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_PREFIX_PATH "$ENV{HOME}/vcpkg/installed/x64-linux/share" ${CMAKE_PREFIX_PATH})

cmake_minimum_required(VERSION 3.14)
project(mlcore_cpp)

set(CMAKE_CXX_STANDARD 17)
set(PYBIND11_FINDPYTHON ON)

find_package(pybind11 REQUIRED)
find_package(xtensor REQUIRED)

include_directories(include)
# add_subdirectory(src)

file(GLOB_RECURSE SRC_FILES src/*.cpp)

pybind11_add_module(mlcore_cpp
    ${SRC_FILES}
)
