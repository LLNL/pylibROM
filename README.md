# pylibROM
Python Interface for LLNL libROM 

## Installation

1. Pull repository and all sub-module dependencies:
  ```
  git clone --recurse-submodules https://github.com/michael-barrow-llnl/pylibROM.git
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


## License
pylibROM is distributed under the terms of the MIT license. All new contributions must be made under the MIT. See
[LICENSE-MIT](https://github.com/LLNL/pylibROM/blob/main/LICENSE)

LLNL Release Nubmer: LLNL-CODE- 852921
