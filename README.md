# pylibROM
Python Interface for LLNL libROM 

## Installation

1. Pull repository and all sub-module dependencies:
  ```
  git clone --recurse-submodules https://github.com/sullan2/pylibROM.git
  ```

2. Compile and build pylibROM (from top-level pylibROM repo):
  ```
  pip install ./
  ```
  To speed up the build if libROM has been compiled:
  ```
  pip install ./ --global-option="--librom_dir=/path/to/pre-installed-libROM"
  ```  
  
3. Test python package (from top-level pylibROM repo):
  ```
  cd tests
  python3.6 testVector.py
  ```

### Using PyMFEM
`pylibROM` is often used together with [`PyMFEM`](https://github.com/mfem/PyMFEM).
Check the repository for detailed instruction for `PyMFEM` installation.
For serial version of `PyMFEM`, the following simple `pip` command works:
```
pip install mfem
```