import sys
import pytest
sys.path.append("../build")

import pylibROM.linalg as libROM
import numpy as np 


# Create an instance of BasisGenerator
options = libROM.Options(4, 20, 3, True, True)
incremental =   False
basis_file_name = "basis.h5"
file_format = libROM.Formats.HDF5
generator = libROM.BasisGenerator(options, incremental, basis_file_name,file_format)

# Test the takeSample method
time = 0.5
dt = 0.5  
u_in_data = np.array([1.0, 2.0, 3.0])
result = generator.takeSample(u_in_data, time, dt)

# Create a BasisWriter instance
base_file_name = "test_basisreader_file"
basis_writer = libROM.BasisWriter(generator, base_file_name, file_format)
basis_writer.writeBasis("basis")

del basis_writer
generator.writeSnapshot()
generator.endSamples()
del generator

# Create an instance of BasisReader
basis_reader = libROM.BasisReader("test_basisreader_file", libROM.Formats.HDF5)

# Test isNewBasis function
is_new_basis = basis_reader.isNewBasis(0.5)
print("isNewBasis: ",is_new_basis)

# Test getSpatialBasis function
spatial_basis1 = basis_reader.getSpatialBasis(0.5)
print("Spatial Basis1: ", spatial_basis1.getData())

# Test getSpatialBasis function
spatial_basis2 = basis_reader.getSpatialBasis(0.5,1)
print("Spatial Basis2: ", spatial_basis2.getData())

# Test getSpatialBasis function
spatial_basis3 = basis_reader.getSpatialBasis(0.5,1,1)
print("Spatial Basis3: ", spatial_basis3.getData())

# Test getSpatialBasis function
spatial_basis4 = basis_reader.getSpatialBasis(0.5,0.7)
print("Spatial Basis4: ", spatial_basis4.getData())

# Test getTemporalBasis function
temporal_basis1 = basis_reader.getTemporalBasis(0.5)
print("Temporal Basis1:", temporal_basis1.getData())

# Test getTemporalBasis function
temporal_basis2 = basis_reader.getTemporalBasis(0.5, 1)
print("Temporal Basis2:",temporal_basis2.getData())

# Test getTemporalBasis function
temporal_basis3 = basis_reader.getTemporalBasis(0.5, 1, 1)
print("Temporal Basis3:",temporal_basis3.getData())

# Test getTemporalBasis function
temporal_basis4 = basis_reader.getTemporalBasis(0.5, 0.7)
print("Temporal Basis4:",temporal_basis4.getData())

# Test getSingularValues function
singular_values1 = basis_reader.getSingularValues(0.5)
print("Singular Values1:", singular_values1.getData())

# Test getSingularValues function
singular_values2 = basis_reader.getSingularValues(0.5, 0.7)
print("Singular Values2:", singular_values2.getData())

# Test getDim function
dim = basis_reader.getDim("basis", 0.5)
print("Dimension:", dim)

# Test getNumSamples function
num_samples = basis_reader.getNumSamples("basis", 0.5)
print("Number of Samples:", num_samples)

# Create an instance of BasisReader with kind-snapshot 
basis_reader1 = libROM.BasisReader("basis.h5_snapshot", libROM.Formats.HDF5)

# Test getSnapshotMatrix function
snapshot_matrix1 = basis_reader1.getSnapshotMatrix(0.5)
print("Snapshot Matrix1:", snapshot_matrix1.getData())

# Test getSnapshotMatrix function
snapshot_matrix2 = basis_reader1.getSnapshotMatrix(0.5, 1)
print("Snapshot Matrix2:", snapshot_matrix2.getData())

# Test getSnapshotMatrix function
snapshot_matrix3 = basis_reader1.getSnapshotMatrix(0.5, 1, 1)
print("Snapshot Matrix3:", snapshot_matrix3.getData())


def test_plus():
    # Create an instance of BasisGenerator
    options = libROM.Options(4, 20, 3, True, True)
    incremental = False
    basis_file_name = "basis.h5"
    file_format = libROM.Formats.HDF5
    generator = libROM.BasisGenerator(options, incremental, basis_file_name, file_format)

    # Test the takeSample method
    time = 0.5
    dt = 0.5  
    u_in_data = np.array([1.0, 2.0, 3.0])
    result = generator.takeSample(u_in_data, time, dt)
    assert result  

    # Create a BasisWriter instance
    base_file_name = "test_basisreader_file"
    basis_writer = libROM.BasisWriter(generator, base_file_name, file_format)
    basis_writer.writeBasis("basis")

    del basis_writer
    generator.writeSnapshot()
    generator.endSamples()
    del generator

    # Create an instance of BasisReader
    basis_reader = libROM.BasisReader("test_basisreader_file", libROM.Formats.HDF5)

    # Test isNewBasis function
    is_new_basis = basis_reader.isNewBasis(0.5)
    assert is_new_basis.getData() == [[-0.2672612419124243], [-0.5345224838248487], [-0.8017837257372731], [-4.4e-323]]   

    # Test getSpatialBasis function
    spatial_basis1 = basis_reader.getSpatialBasis(0.5)
    assert spatial_basis1.getData() == [[-0.2672612419124243], [-0.5345224838248487], [-0.8017837257372731], [-4.4e-323]]  

    # Test getSpatialBasis function
    spatial_basis2 = basis_reader.getSpatialBasis(0.5, 1)
    assert spatial_basis2.getData() ==  [[-0.2672612419124243], [-0.5345224838248487], [-0.8017837257372731], [-4.4e-323]]

    # Test getSpatialBasis function
    spatial_basis3 = basis_reader.getSpatialBasis(0.5, 1, 1)
    assert spatial_basis3.getData() ==  [[-0.2672612419124243], [-0.5345224838248487], [-0.8017837257372731], [-4.4e-323]] 

    # Test getSpatialBasis function
    spatial_basis4 = basis_reader.getSpatialBasis(0.5, 0.7)
    assert spatial_basis4.getData() ==  [[-1.0]]

    # Test getTemporalBasis function
    temporal_basis1 = basis_reader.getTemporalBasis(0.5)
    assert temporal_basis1.getData() == [[-1.0]]  

    # Test getTemporalBasis function
    temporal_basis2 = basis_reader.getTemporalBasis(0.5, 1)
    assert temporal_basis2.getData() ==  [[-1.0]] 

    # Test getTemporalBasis function
    temporal_basis3 = basis_reader.getTemporalBasis(0.5, 1, 1)
    assert temporal_basis3.getData() ==  [[-1.0]]

    # Test getTemporalBasis function
    temporal_basis4 = basis_reader.getTemporalBasis(0.5, 0.7)
    assert temporal_basis4.getData() ==  [[-1.0]] 

    # Test getSingularValues function
    singular_values1 = basis_reader.getSingularValues(0.5)
    assert singular_values1.getData() ==  [3.7416573867739418] 

    # Test getSingularValues function
    singular_values2 = basis_reader.getSingularValues(0.5, 0.7)
    assert singular_values2.getData() == [3.7416573867739418]  

    # Test getDim function
    dim = basis_reader.getDim("basis", 0.5)
    assert   dim == 4

    # Test getNumSamples function
    num_samples = basis_reader.getNumSamples("basis", 0.5)
    assert num_samples == 1

    # Create an instance of BasisReader with kind-snapshot 
    basis_reader1 = libROM.BasisReader("basis.h5_snapshot", libROM.Formats.HDF5)

    # Test getSnapshotMatrix function
    snapshot_matrix1 = basis_reader1.getSnapshotMatrix(0.5)
    assert snapshot_matrix1.getData()== [[-3.7416573867739418], [0.4217934441190679], [0.6326901661786019], [3.5e-323]] 

    # Test getSnapshotMatrix function
    snapshot_matrix2 = basis_reader1.getSnapshotMatrix(0.5, 1)
    assert snapshot_matrix2.getData() == [[-3.7416573867739418], [0.4217934441190679], [0.6326901661786019], [3.5e-323]]

    # Test getSnapshotMatrix function
    snapshot_matrix3 = basis_reader1.getSnapshotMatrix(0.5, 1, 1)
    assert snapshot_matrix3.getData() == [[-3.7416573867739418], [0.4217934441190679], [0.6326901661786019], [3.5e-323]]

if __name__ == "__main__":
    pytest.main()
