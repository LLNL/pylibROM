FROM ubuntu:22.04

ENV ENVDIR=env

# install sudo
RUN apt-get -yq update && apt-get -yq install sudo

WORKDIR /$ENVDIR

# install packages
RUN sudo apt-get install -yq git
RUN sudo apt-get install --no-install-recommends -yq make gcc gfortran libssl-dev cmake
RUN sudo apt-get install -yq libopenblas-dev libmpich-dev libblas-dev liblapack-dev libscalapack-mpi-dev libhdf5-mpi-dev hdf5-tools
RUN sudo apt-get install -yq vim
RUN sudo apt-get install -yq git-lfs
RUN sudo apt-get install -yq valgrind
RUN sudo apt-get install -yq wget
RUN sudo apt-get install -yq astyle
RUN sudo apt-get install -yq chrpath

RUN sudo apt-get clean -q

# download dependencies
ENV LIB_DIR=/$ENVDIR/dependencies
WORKDIR $LIB_DIR

# install googletest
WORKDIR $LIB_DIR
RUN git clone https://github.com/google/googletest
WORKDIR ./googletest
# Last release that supports c++11
RUN git checkout tags/release-1.12.1 -b v1.12.1
WORKDIR ./build
RUN cmake ..
RUN make
RUN sudo make install

# install python
RUN sudo apt-get update
RUN sudo apt-get install -yq python3
RUN sudo apt-get install -yq python3-dev
RUN sudo apt-get install -yq python3-pip

RUN echo "numpy" >> requirements.txt
RUN echo "scipy" >> requirements.txt
RUN echo "argparse" >> requirements.txt
RUN echo "tables" >> requirements.txt
RUN echo "PyYAML" >> requirements.txt
RUN echo "h5py" >> requirements.txt
RUN echo "pybind11" >> requirements.txt
RUN echo "pytest" >> requirements.txt
RUN echo "mpi4py" >> requirements.txt
RUN sudo pip3 install --upgrade pip
RUN sudo pip3 install -r ./requirements.txt

# install pymfem
RUN sudo pip3 install swig

WORKDIR $LIB_DIR
RUN git clone https://github.com/mfem/PyMFEM.git
WORKDIR ./PyMFEM
# --with-gslib flag does not work with later versions.
RUN git checkout v_4.7.0.1
# We need to use this hack until we can submit a pull request to PyMFEM.
#RUN wget -O setup.py https://github.com/LLNL/pylibROM/raw/prom_parallel/extern/PyMFEM.setup.py
RUN sudo python3 setup.py install --with-parallel --with-gslib --with-lapack

# install lldb and gdb for debugging
RUN sudo apt-get install -yq lldb gdb

# create and switch to a user
ENV USERNAME=test
RUN echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
RUN useradd --no-log-init -u 1001 --create-home --shell /bin/bash $USERNAME
RUN adduser $USERNAME sudo
USER $USERNAME
WORKDIR /home/$USERNAME
