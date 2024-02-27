# pylibROM
Python Interface for LLNL libROM 

## Installation

1. Pull repository and all sub-module dependencies:
  ```
  git clone --recurse-submodules https://github.com/llnl/pylibROM.git
  ```

2. Compile and build pylibROM (from top-level pylibROM repo):
  ```
  pip install ./
  ```
  To speed up the build if libROM has been compiled:
  ```
  pip install ./ --global-option="--librom_dir=/path/to/pre-installed-libROM"
  ```  
  If you want to build static ScaLAPACK for libROM,
  ```
  pip install ./ --global-option="--install_scalapack"
  ```
  
3. Test python package (from top-level pylibROM repo):
  ```
  cd tests
  pytest test_pyVector.py
  ```

### Using PyMFEM
`pylibROM` is often used together with [`PyMFEM`](https://github.com/mfem/PyMFEM).
Check the repository for detailed instruction for `PyMFEM` installation.
For serial version of `PyMFEM`, the following simple `pip` command works:
```
pip install mfem
```
For parallel version, a manual installation is required:
```
git clone https://github.com/mfem/PyMFEM.git
cd PyMFEM
python3 setup.py install --with-parallel --with-gslib
```
On LC quartz, use `--user` flag:
```
python3 setup.py install --with-parallel --with-gslib --user
```
Make sure [`swig`](https://pypi.org/project/swig) is installed first. Also, the binary file must be located in `PATH` environment variable.

### pylibROM-Jupyter Docker Container

This Docker container provides an environment with Jupyter Notebook for the pylibROM library. Follow the steps below to build the Docker image and run a Jupyter Notebook server.

#### Build the Docker Image

Navigate to the directory containing the Dockerfile:
 ```
cd /path/to/folder/pylibROM/docker/jupyter
 ```

Now, run the following command to build the Docker image:
 
 ```
docker build -t pylibrom-jupyter:latest .
 ```

Once the image is built, you can run a container and start a Jupyter Notebook server. Replace /path/to/host/folder with the absolute path to the local directory you want to mount inside the container for Jupyter notebooks: 

 ```
docker run -p 8888:8888 -v /path/to/host/folder:/notebooks -w /notebooks pylibrom-jupyter:latest
 ```


## License
pylibROM is distributed under the terms of the MIT license. All new contributions must be made under the MIT. See
[LICENSE-MIT](https://github.com/LLNL/pylibROM/blob/main/LICENSE)

LLNL Release Nubmer: LLNL-CODE- 852921


## Authors
- Michael Barrow (LLNL)
- Siu Wun Cheung (LLNL)
- Youngsoo Choi (LLNL)
- "Kevin" Seung Whan Chung (LLNL)
- Pravija Danda (TAMU)
- Coleman Kendrick (LLNL)
- Hardeep Sullan (LLNL)
- Jian Tao (TAMU)
- Henry Yu (LLNL)
