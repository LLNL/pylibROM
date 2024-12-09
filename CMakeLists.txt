cmake_minimum_required(VERSION 3.12)

project(_pylibROM)

set(CMAKE_BUILD_TYPE Debug)
set(PYBIND11_FINDPYTHON ON)

option (USE_MFEM "Build pylibROM with MFEM" OFF)

#=================== ScaLAPACK (optional) ==================
option(BUILD_SCALAPACK "Build static ScaLAPACK for libROM" OFF)

if (BUILD_SCALAPACK)
	set(WORK_DIR ${CMAKE_CURRENT_SOURCE_DIR}/extern)
	add_custom_command(
		OUTPUT SLPK_BUILD
		WORKING_DIRECTORY ${WORK_DIR}
		COMMAND ${WORK_DIR}/librom_scalapack.sh
		COMMENT "Building Static ScaLAPACK..."
	)
	add_custom_target(RUN_SLPK_BUILD ALL DEPENDS SLPK_BUILD)
endif(BUILD_SCALAPACK)

# HDF5
find_package(HDF5 1.8.0 REQUIRED)

#=================== libROM ================================

#It is tedious to build libROM. option to not build
set(LIBROM_DIR "" CACHE STRING "absolute path to the pre-installed libROM")

set(BUILD_LIBROM OFF)
if (NOT LIBROM_DIR)
  set(LIBROM_DIR ${CMAKE_CURRENT_SOURCE_DIR}/extern/libROM)
  set(LIBROM_SCRIPTS_DIR ${LIBROM_DIR}/scripts)
  set(BUILD_LIBROM ON)
endif()

include(ExternalProject)

include(CMakePrintHelpers)
cmake_print_variables(LIBROM_DIR)
cmake_print_variables(LIBROM_SCRIPTS_DIR)

if (BUILD_LIBROM)
	# add_custom_command(
	# 	OUTPUT LIBROM_BUILD
	# 	WORKING_DIRECTORY ${LIBROM_DIR}
	# 	COMMAND ${LIBROM_SCRIPTS_DIR}/compile.sh
	# 	ARGS -m -t ${LIBROM_DIR}/cmake/toolchains/simple.cmake
	# 	COMMENT "Building libROM..."
	# )
	# add_custom_target(RUN_LIBROM_BUILD ALL DEPENDS LIBROM_BUILD)

    set(LIBROM_BUILD_CMD "${LIBROM_SCRIPTS_DIR}/compile.sh -t ${LIBROM_DIR}/cmake/toolchains/simple.cmake" CACHE STRING  "Command used to build libROM and dependencies")
    if (USE_MFEM)
        set(LIBROM_BUILD_CMD "${LIBROM_BUILD_CMD} -m -g -l")
    endif()
	# ExternalProject_Add(
    #     	libROM
    #     	SOURCE_DIR ${LIBROM_SCRIPTS_DIR}
    #     	CONFIGURE_COMMAND ""
    #     	BINARY_DIR ${LIBROM_DIR}
    #         BUILD_COMMAND ${LIBROM_BUILD_CMD}
    #         INSTALL_COMMAND ""
    # )
	message("Building libROM dependency...")
endif(BUILD_LIBROM)

#setup external dependency build and link paths for libROM
set(LIBROM_INCLUDE_DIR ${LIBROM_DIR}/lib)
link_directories(${LIBROM_DIR}/build/lib) #this hack is the best way for non-cmake since FetchContent will not work for compile.sh

#include mpi4py directory
execute_process(COMMAND python3 -c "import mpi4py; print(mpi4py.get_include())" OUTPUT_VARIABLE MPI4PY)

# # MFEM is required.
# # TODO(kevin): We do not bind mfem-related functions until we figure out how to type-cast SWIG Object.
# #              Until then, mfem-related functions need to be re-implemented on python-end, using PyMFEM.

if (USE_MFEM)
    find_library(MFEM mfem
        "$ENV{MFEM_DIR}/lib"
        "$ENV{MFEM_DIR}"
        "${LIBROM_DIR}/dependencies/mfem")
    find_library(HYPRE HYPRE
        "$ENV{HYPRE_DIR}/lib"
        "${LIBROM_DIR}/dependencies/hypre/src/hypre/lib")
    find_library(PARMETIS parmetis
        "$ENV{PARMETIS_DIR}/lib"
        "$ENV{PARMETIS_DIR}/build/lib/libparmetis"
        "${LIBROM_DIR}/dependencies/parmetis-4.0.3/build/lib/libparmetis")
    find_library(METIS metis
        "$ENV{METIS_DIR}/lib"
        "$ENV{PARMETIS_DIR}/build/lib/libmetis"
        "${LIBROM_DIR}/dependencies/parmetis-4.0.3/build/lib/libmetis")
    find_path(MFEM_INCLUDES mfem.hpp
        "$ENV{MFEM_DIR}/include"
        "$ENV{MFEM_DIR}"
        "${LIBROM_DIR}/dependencies/mfem")
    find_path(HYPRE_INCLUDES HYPRE.h
        "$ENV{HYPRE_DIR}/include"
        "${LIBROM_DIR}/dependencies/hypre/src/hypre/include")
    find_path(PARMETIS_INCLUDES metis.h
        "$ENV{PARMETIS_DIR}/metis/include"
        "${LIBROM_DIR}/dependencies/parmetis-4.0.3/metis/include")
    set(PYLIBROM_HAS_MFEM 1)
endif()
#===================== pylibROM =============================


set(CMAKE_CXX_STANDARD 14)

find_package(MPI REQUIRED)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/bindings/pylibROM/pylibROM_config.h.in
               ${CMAKE_CURRENT_SOURCE_DIR}/bindings/pylibROM/pylibROM_config.h)

set(SOURCE_DIR "bindings/pylibROM")
include_directories(
	${SOURCE_DIR} 
	${LIBROM_INCLUDE_DIR} 
	${MPI_INCLUDE_PATH}
	${MPI4PY}
	${HDF5_C_INCLUDE_DIRS}
)
link_libraries(${HDF5_LIBRARIES})

add_subdirectory("extern/pybind11")

set(PYLIBROM_SOURCES
	bindings/pylibROM/pylibROM.cpp

	bindings/pylibROM/linalg/pyMatrix.cpp
	bindings/pylibROM/linalg/pyVector.cpp
	bindings/pylibROM/linalg/pyBasisGenerator.cpp
	bindings/pylibROM/linalg/pyBasisReader.cpp
	bindings/pylibROM/linalg/pyBasisWriter.cpp
	bindings/pylibROM/linalg/pyOptions.cpp
	bindings/pylibROM/linalg/pyNNLS.cpp
	bindings/pylibROM/linalg/svd/pySVD.cpp
	bindings/pylibROM/linalg/svd/pyStaticSVD.cpp
	bindings/pylibROM/linalg/svd/pyIncrementalSVD.cpp

	bindings/pylibROM/algo/pyDMD.cpp
	bindings/pylibROM/algo/pyParametricDMD.cpp
	bindings/pylibROM/algo/pyNonuniformDMD.cpp
	bindings/pylibROM/algo/pyAdaptiveDMD.cpp
	bindings/pylibROM/algo/greedy/pyGreedySampler.cpp
	bindings/pylibROM/algo/greedy/pyGreedyCustomSampler.cpp
	bindings/pylibROM/algo/greedy/pyGreedyRandomSampler.cpp
	bindings/pylibROM/algo/manifold_interp/pyInterpolator.cpp
	bindings/pylibROM/algo/manifold_interp/pyMatrixInterpolator.cpp
	bindings/pylibROM/algo/manifold_interp/pyVectorInterpolator.cpp

	bindings/pylibROM/hyperreduction/pyDEIM.cpp
	bindings/pylibROM/hyperreduction/pyGNAT.cpp
	bindings/pylibROM/hyperreduction/pyQDEIM.cpp
	bindings/pylibROM/hyperreduction/pyS_OPT.cpp
	bindings/pylibROM/hyperreduction/pySTSampling.cpp
	bindings/pylibROM/hyperreduction/pyUtilities.cpp

	bindings/pylibROM/utils/mpi_utils.cpp
	bindings/pylibROM/utils/pyDatabase.hpp
	bindings/pylibROM/utils/pyDatabase.cpp
	bindings/pylibROM/utils/pyHDFDatabase.cpp
	bindings/pylibROM/utils/pyHDFDatabaseMPIO.cpp
	bindings/pylibROM/utils/pyCSVDatabase.cpp

	bindings/pylibROM/python_utils/cpp_utils.hpp
)

if (USE_MFEM)
    set(PYLIBROM_SOURCES ${PYLIBROM_SOURCES}
        bindings/pylibROM/mfem/pyUtilities.cpp
        bindings/pylibROM/mfem/pyPointwiseSnapshot.cpp
        bindings/pylibROM/mfem/pySampleMesh.cpp)
endif()

pybind11_add_module(_pylibROM ${PYLIBROM_SOURCES})
message("building pylibROM...")

if (USE_MFEM)
    target_include_directories(
        _pylibROM
        PUBLIC
        ${MFEM_INCLUDES}
        ${HYPRE_INCLUDES}
        ${PARMETIS_INCLUDES}
        ${MFEM_C_INCLUDE_DIRS})

    target_link_libraries(
        _pylibROM
        PUBLIC
        ${MFEM}
        ${HYPRE}
        ${PARMETIS}
        ${METIS})
endif()

target_link_libraries(_pylibROM PRIVATE ROM)
