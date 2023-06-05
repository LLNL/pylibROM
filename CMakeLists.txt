cmake_minimum_required(VERSION 3.12)

#setup external dependency build and link paths for libROM
set(LIBROM_SCRIPTS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/extern/libROM/scripts)
set(LIBROM_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/extern/libROM/lib)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/extern/libROM/build/lib) #this hack is the best way for non-cmake since FetchContent will not work for compile.sh

include(ExternalProject)

#It is tedious to build libROM. option to not build
option(BUILD_DEPS "compile libROM library " ON)
if (BUILD_DEPS)
	set (LIBROM libROM)
	ExternalProject_Add(
        	${LIBROM}
        	SOURCE_DIR ${LIBROM_SCRIPTS_DIR}
        	CONFIGURE_COMMAND ""
        	BINARY_DIR ${LIBROM_SCRIPTS_DIR}
        	BUILD_COMMAND ${LIBROM_SCRIPTS_DIR}/compile.sh
        	INSTALL_COMMAND ""
	)	
	message("Building libROM dependency...")
endif(BUILD_DEPS)

project(pylibROM)

#link_directories("extern/libROM/build/lib")

set(CMAKE_CXX_STANDARD 14)


set(SOURCE_DIR "bindings/pylibROM") 
include_directories(${SOURCE_DIR} ${LIBROM_INCLUDE_DIR})
add_subdirectory("extern/pybind11")


# pybind11_add_module(pylibROM ${SOURCE_DIR} "${SOURCE_DIR}/linalg/pyVector.cpp")
pybind11_add_module(pylibROM
    bindings/pylibROM/linalg.cpp
    bindings/pylibROM/linalg/pyMatrix.cpp
    bindings/pylibROM/linalg/pyVector.cpp
)

target_link_libraries(pylibROM PRIVATE ROM) 




