#!/bin/bash
check_result () {
  # $1: Result output of the previous command ($?)
  # $2: Name of the previous command
  if [ $1 -eq 0 ]; then
      echo "$2 succeeded"
  else
      echo "$2 failed"
      exit -1
  fi
}

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
LIB_DIR=$SCRIPT_DIR/libROM/dependencies
mkdir -p $LIB_DIR

# Install ScaLAPACK if specified.
cd $LIB_DIR
if [ -f "scalapack-2.2.0/libscalapack.a" ]; then
  echo "Using dependencies/scalapack-2.2.0/libscalapack.a"
else
  echo "ScaLAPACK is needed!"
  tar -zxvf scalapack-2.2.0.tar.gz
  cp SLmake.inc scalapack-2.2.0/
  cd scalapack-2.2.0/
  make
  check_result $? ScaLAPACK-installation
fi
