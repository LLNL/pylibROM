# pylibROM
## Python Interface for LLNL libROM 

1. Build libROM library (from top-level pylibROM repo) Defunct:
  ```
  cd extern/libROM
  .scripts/compile.sh -m
  ```

2. Compile and build pylibROM (from top-level pylibROM repo):
  ```
  mkdir build
  cd build
  cmake ..
  make
  ```
  
3. Test python package (from top-level pylibROM repo):
  ```
  cd tests
  python3.6 testVector.py
  ```

