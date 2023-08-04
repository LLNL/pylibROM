import sys
import sys
import pytest
sys.path.append("../build")
import pylibROM.linalg as libROM
import numpy as np 

def test_writeBasis():
    options = libROM.Options(4, 20, 3, True, True)
    generator = libROM.BasisGenerator(options, False, "basis.h5", libROM.Formats.HDF5)
    result = generator.takeSample(np.array([1.0, 2.0, 3.0]), 1.0, 0.1) 
    base_file_name = "basis_file"
    basis_writer = libROM.BasisWriter(generator, base_file_name, libROM.Formats.HDF5)
    basis_writer.writeBasis("basis")
   
if __name__ == "__main__":
    pytest.main()




