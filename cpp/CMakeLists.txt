set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

cmake_minimum_required(VERSION 3.14)
project(mlcore_cpp)

set(CMAKE_CXX_STANDARD 17)

find_package(pybind11 REQUIRED)

include_directories(include)
# add_subdirectory(src)

pybind11_add_module(mlcore_cpp
    example.cpp
    # src/linear_regression.cpp
)
