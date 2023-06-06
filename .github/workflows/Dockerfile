FROM dreamer2368/librom_env:v0.3.1

# install libROM for scaleupROM
WORKDIR $LIB_DIR
RUN sudo git clone https://github.com/LLNL/libROM.git
WORKDIR ./libROM
# some minor revisions needed for scaleupROM, not merged yet into master branch.
RUN sudo git pull
WORKDIR ./build
RUN sudo cmake .. -DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_FILE} -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DUSE_MFEM=${USE_MFEM} -DMFEM_USE_GSLIB=${MFEM_USE_GSLIB}
RUN sudo make -j 16

ENV LIBROM_DIR=$LIB_DIR/libROM

# create and switch to a user
WORKDIR /home/$USERNAME
