cmake_minimum_required(VERSION 3.12)
project(pylibROM)

set(CMAKE_CXX_STANDARD 14)

set(SOURCE_DIR "bindings/pylibROM")
include_directories(${SOURCE_DIR})

include_directories("extern/libROM/lib/utils")
include_directories("extern/libROM/lib")

#add_subdirectory("extern/libROM")
add_subdirectory("extern/pybind11")


pybind11_add_module(pylibROM ${SOURCE_DIR} "${SOURCE_DIR}/linalg/pyVector.cpp")
#target_link_libraries(pylibROM PRIVATE ROM)
