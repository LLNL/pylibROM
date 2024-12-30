import sys
import pytest
try:
    # import pip-installed package
    import pylibROM
    import pylibROM.linalg as libROM
    from pylibROM.utils import Database
except ModuleNotFoundError:
    # If pip-installed package is not found, import cmake-built package
    sys.path.append("../build")
    import _pylibROM as pylibROM
    import _pylibROM.linalg as libROM
    from _pylibROM.utils import Database
import numpy as np 

def test_writeBasis():
    options = libROM.Options(4, 20, True, True)
    generator = libROM.BasisGenerator(options, False, "basis.h5", Database.formats.HDF5)
    result = generator.takeSample(np.array([1.0, 2.0, 3.0])) 
    base_file_name = "basis_file"
    basis_writer = libROM.BasisWriter(generator, base_file_name, Database.formats.HDF5)
    basis_writer.writeBasis("basis")
   
if __name__ == "__main__":
    pytest.main()




