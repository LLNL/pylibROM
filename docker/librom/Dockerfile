FROM ghcr.io/llnl/librom/librom_env:latest

RUN sudo apt-get -yq update
RUN sudo apt-get install -yq chrpath

# Add any more dependencies you would like to have in ci test/development environment.
# RUN python3 -m pip install numpy
RUN sudo pip3 install --upgrade pip
RUN sudo pip3 install swig

WORKDIR $LIB_DIR
RUN sudo git clone https://github.com/mfem/PyMFEM.git
WORKDIR ./PyMFEM
# --with-gslib flag does not work with later versions.
RUN sudo git checkout v4.5.2
# We need to use this hack until we can submit a pull request to PyMFEM.
RUN sudo wget -O setup.py https://github.com/LLNL/pylibROM/raw/prom_parallel/extern/PyMFEM.setup.py
RUN sudo python3 setup.py install --with-parallel --with-gslib

# install libROM
WORKDIR $LIB_DIR
RUN sudo git clone https://github.com/LLNL/libROM.git
WORKDIR ./libROM
RUN sudo git pull
# pylibROM is currently based on a specific commit of the libROM.
RUN sudo git checkout 0809d7d09dc24f0963c38fc8c0a2649948142ba0
WORKDIR ./build
RUN sudo cmake .. -DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_FILE} -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DUSE_MFEM=${USE_MFEM} -DMFEM_USE_GSLIB=${MFEM_USE_GSLIB}
# RUN sudo cmake .. -DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_FILE} -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DUSE_MFEM=OFF
RUN sudo make -j 16

# ENV LIBROM_DIR=$LIB_DIR/libROM

# create and switch to a user
WORKDIR /home/$USERNAME
