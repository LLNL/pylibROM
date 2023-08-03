cmake_minimum_required(VERSION 3.12)

project(pylibROM)

set(CMAKE_BUILD_TYPE Debug)
set(PYBIND11_FINDPYTHON ON)

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
	# add_custom_command(
	# 	OUTPUT LIBROM_BUILD
	# 	WORKING_DIRECTORY ${LIBROM_DIR}
	# 	COMMAND ${LIBROM_SCRIPTS_DIR}/compile.sh
	# 	ARGS -m -t ${LIBROM_DIR}/cmake/toolchains/simple.cmake
	# 	COMMENT "Building libROM..."
	# )
	# add_custom_target(RUN_LIBROM_BUILD ALL DEPENDS LIBROM_BUILD)

	ExternalProject_Add(
        	libROM
        	SOURCE_DIR ${LIBROM_SCRIPTS_DIR}
        	CONFIGURE_COMMAND ""
        	BINARY_DIR ${LIBROM_DIR}
        	BUILD_COMMAND ${LIBROM_SCRIPTS_DIR}/compile.sh -t ${LIBROM_DIR}/cmake/toolchains/simple.cmake
        	INSTALL_COMMAND ""
	)	
	message("Building libROM dependency...")
endif(BUILD_DEPS)

#setup external dependency build and link paths for libROM
set(LIBROM_INCLUDE_DIR ${LIBROM_DIR}/lib)
link_directories(${LIBROM_DIR}/build/lib) #this hack is the best way for non-cmake since FetchContent will not work for compile.sh

#include mpi4py directory
execute_process(COMMAND python3 -c "import mpi4py; print(mpi4py.get_include())" OUTPUT_VARIABLE MPI4PY)

# MFEM is required.
find_library(MFEM mfem "$ENV{MFEM_DIR}/lib" "${LIBROM_DIR}/dependencies/mfem/lib")
find_library(HYPRE HYPRE "$ENV{HYPRE_DIR}/lib" "${LIBROM_DIR}/dependencies/hypre/lib")
find_library(PARMETIS parmetis "$ENV{PARMETIS_DIR}/lib" "${LIBROM_DIR}/dependencies/parmetis-4.0.3/build/lib/libparmetis/lib")
find_library(METIS metis "$ENV{METIS_DIR}/lib" "${LIBROM_DIR}/dependencies/parmetis-4.0.3/build/lib/libmetis/lib")
find_path(MFEM_INCLUDES mfem.hpp "$ENV{MFEM_DIR}/include" "${LIBROM_DIR}/dependencies/mfem/include")
find_path(HYPRE_INCLUDES HYPRE.h "$ENV{HYPRE_DIR}/include" "${LIBROM_DIR}/dependencies/hypre/include")
find_path(PARMETIS_INCLUDES metis.h "$ENV{PARMETIS_DIR}/metis/include" "${LIBROM_DIR}/dependencies/parmetis-4.0.3/build/lib/libparmetis/metis/include")

#===================== pylibROM =============================


set(CMAKE_CXX_STANDARD 14)

find_package(MPI REQUIRED)

set(SOURCE_DIR "bindings/pylibROM") 
include_directories(
	${SOURCE_DIR} 
	${LIBROM_INCLUDE_DIR} 
	${MPI_INCLUDE_PATH}
	${MPI4PY}
	${MFEM_INCLUDES}
	${HYPRE_INCLUDES}
	${PARMETIS_INCLUDES}
	${MFEM_C_INCLUDE_DIRS}
)
link_libraries(
	${MFEM}
	${HYPRE}
	${PARMETIS}
	${METIS}
)

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
	bindings/pylibROM/utils/mpi_utils.cpp
	bindings/pylibROM/utils/pyDatabase.cpp

	bindings/pylibROM/mfem/Utilities.cpp
)
message("building pylibROM...")

target_link_libraries(pylibROM PRIVATE ROM)
