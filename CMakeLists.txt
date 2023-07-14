cmake_minimum_required(VERSION 3.12)

set(CMAKE_BUILD_TYPE Debug)

#=================== libROM ================================

#It is tedious to build libROM. option to not build
set(LIBROM_DIR "" CACHE STRING "absolute path to the pre-installed libROM")
set(BUILD_DEPS OFF)
if (NOT LIBROM_DIR)
  set(LIBROM_DIR ${CMAKE_CURRENT_SOURCE_DIR}/extern/libROM)
  set(LIBROM_SCRIPTS_DIR ${LIBROM_DIR}/scripts)
  set(BUILD_DEPS ON)
endif()

include(ExternalProject)

include(CMakePrintHelpers)
cmake_print_variables(LIBROM_DIR)
cmake_print_variables(LIBROM_SCRIPTS_DIR)

if (BUILD_DEPS)
	set (LIBROM libROM)
	ExternalProject_Add(
        	${LIBROM}
        	SOURCE_DIR ${LIBROM_SCRIPTS_DIR}
        	CONFIGURE_COMMAND ""
        	BINARY_DIR ${LIBROM_SCRIPTS_DIR}
        	BUILD_COMMAND ${LIBROM_SCRIPTS_DIR}/compile.sh -d
        	INSTALL_COMMAND ""
	)	
	message("Building libROM dependency...")
endif(BUILD_DEPS)

#setup external dependency build and link paths for libROM
set(LIBROM_INCLUDE_DIR ${LIBROM_DIR}/lib)
link_directories(${LIBROM_DIR}/build/lib) #this hack is the best way for non-cmake since FetchContent will not work for compile.sh

#===================== pylibROM =============================
project(pylibROM)

set(CMAKE_CXX_STANDARD 14)

find_package(MPI REQUIRED)

set(SOURCE_DIR "bindings/pylibROM") 
include_directories(	${SOURCE_DIR} 
			${LIBROM_INCLUDE_DIR} 
			${MPI_INCLUDE_PATH}	)
add_subdirectory("extern/pybind11")


pybind11_add_module(pylibROM
    bindings/pylibROM/pylibROM.cpp
    bindings/pylibROM/linalg/pyMatrix.cpp
    bindings/pylibROM/linalg/pyVector.cpp
    bindings/pylibROM/linalg/pyBasisGenerator.cpp
    bindings/pylibROM/linalg/pyOptions.cpp
    bindings/pylibROM/algo/pyDMD.cpp
)
message("building pylibROM...")

target_link_libraries(pylibROM PRIVATE ROM) 
