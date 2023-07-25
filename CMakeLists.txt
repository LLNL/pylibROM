cmake_minimum_required(VERSION 3.12)

set(CMAKE_BUILD_TYPE Debug)

#=================== libROM ================================

#setup external dependency build and link paths for libROM
set(LIBROM_SCRIPTS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/extern/libROM/scripts)
set(LIBROM_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/extern/libROM/lib)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/extern/libROM/build/lib) #this hack is the best way for non-cmake since FetchContent will not work for compile.sh

include(ExternalProject)

#It is tedious to build libROM. option to not build
# option(BUILD_DEPS "compile libROM library " ON)
# if (BUILD_DEPS)
# 	set (LIBROM libROM)
# 	ExternalProject_Add(
#         	${LIBROM}
#         	SOURCE_DIR ${LIBROM_SCRIPTS_DIR}
#         	CONFIGURE_COMMAND ""
#         	BINARY_DIR ${LIBROM_SCRIPTS_DIR}
#         	BUILD_COMMAND ${LIBROM_SCRIPTS_DIR}/compile.sh -d
#         	INSTALL_COMMAND ""
# 	)	
# 	message("Building libROM dependency...")
# endif(BUILD_DEPS)

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
    bindings/pylibROM/linalg/pyBasisReader.cpp
	bindings/pylibROM/linalg/pyBasisWriter.cpp
    bindings/pylibROM/linalg/pyOptions.cpp
    bindings/pylibROM/linalg/svd/pySVD.cpp
    bindings/pylibROM/linalg/svd/pyStaticSVD.cpp
    bindings/pylibROM/linalg/svd/pyIncrementalSVD.cpp
    bindings/pylibROM/algo/pyDMD.cpp 
)
message("building pylibROM...")

target_link_libraries(pylibROM PRIVATE ROM) 