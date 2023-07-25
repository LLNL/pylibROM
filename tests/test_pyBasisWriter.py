import sys
import pytest
sys.path.append("..")

import build.pylibROM.linalg as libROM
import numpy as np 

# Create an instance of BasisGenerator
options = libROM.Options(4, 20, 3, True, True)
incremental =   False
basis_file_name = "basis.h5"
file_format = libROM.Formats.HDF5
generator = libROM.BasisGenerator(options, incremental, basis_file_name,file_format)

# Test the takeSample method
time = 1.0  # Time of the state
dt = 0.1  # Time step
u_in_data = np.array([1.0, 2.0, 3.0])
result = generator.takeSample(u_in_data, time, dt)

# Create a BasisWriter instance
base_file_name = "basis_file"
basis_writer = libROM.BasisWriter(generator, base_file_name, file_format)

# Test the writeBasis method
basis_writer.writeBasis("basis")

def test_plus():
    # Create an instance of BasisGenerator
    options = libROM.Options(4, 20, 3, True, True)
    incremental = False
    basis_file_name = "basis.h5"
    file_format = libROM.Formats.HDF5
    generator = libROM.BasisGenerator(options, incremental, basis_file_name, file_format)

    # Test the takeSample method
    time = 1.0  # Time of the state
    dt = 0.1  # Time step
    u_in_data = np.array([1.0, 2.0, 3.0])
    result = generator.takeSample(u_in_data, time, dt)
    assert result  

    # Create a BasisWriter instance
    base_file_name = "basis_file"
    basis_writer = libROM.BasisWriter(generator, base_file_name, file_format)

    # Test the writeBasis method
    basis_writer.writeBasis("basis")
   

if __name__ == "__main__":
    pytest.main()




