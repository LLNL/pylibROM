FROM ghcr.io/llnl/librom/librom_env:latest

# Add any more dependencies you would like to have in ci test/development environment.
# RUN python3 -m pip install numpy

# install libROM
WORKDIR $LIB_DIR
RUN sudo git clone https://github.com/LLNL/libROM.git
WORKDIR ./libROM
RUN sudo git pull
# pylibROM is currently based on a specific commit of the libROM.
RUN sudo git checkout 6f341200ec0ef03b84d6b7bbe4d338ed3c72a8b1
WORKDIR ./build
RUN sudo cmake .. -DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_FILE} -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DUSE_MFEM=${USE_MFEM} -DMFEM_USE_GSLIB=${MFEM_USE_GSLIB}
RUN sudo make -j 16

# ENV LIBROM_DIR=$LIB_DIR/libROM

# create and switch to a user
WORKDIR /home/$USERNAME
