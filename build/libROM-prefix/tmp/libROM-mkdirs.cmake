# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/g/g92/yu39/git/pylibROM/extern/libROM/scripts"
  "/g/g92/yu39/git/pylibROM/extern/libROM/scripts"
  "/g/g92/yu39/git/pylibROM/build/libROM-prefix"
  "/g/g92/yu39/git/pylibROM/build/libROM-prefix/tmp"
  "/g/g92/yu39/git/pylibROM/build/libROM-prefix/src/libROM-stamp"
  "/g/g92/yu39/git/pylibROM/build/libROM-prefix/src"
  "/g/g92/yu39/git/pylibROM/build/libROM-prefix/src/libROM-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/g/g92/yu39/git/pylibROM/build/libROM-prefix/src/libROM-stamp/${subDir}")
endforeach()
