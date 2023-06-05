# pylibROM
## Python Interface for LLNL libROM 

1. Pull repository and all sub-module dependencies:
  ```
  git clone --recurse-submodules https://github.com/sullan2/pylibROM.git
  ```

2. Compile and build pylibROM (from top-level pylibROM repo):
  ```
  mkdir build
  cd build
  cmake ..
  make
  ```
  To speed up the build if libROM has been compiled:
  ```
  cmake .. -DBUILD_DEPS=OFF #Do not build libROM
  ```  
  
3. Test python package (from top-level pylibROM repo):
  ```
  cd tests
  python3.6 testVector.py
  ```

