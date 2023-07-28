import sys
import pytest
sys.path.append("../build")

import pylibROM.linalg as libROM
import numpy as np 
import h5py

# Create an instance of BasisGenerator
options = libROM.Options(4, 20, 3, True, True)
incremental =   False
basis_file_name = "basis.h5"
file_format = libROM.Formats.HDF5
generator = libROM.BasisGenerator(options, incremental, basis_file_name,file_format)

# Test the isNextSample method
time = 1.0  # Time of interest
is_next = generator.isNextSample(time)
print("isNextSample:", is_next)

# Test the updateRightSV method
update_success = generator.updateRightSV()
print("updateRightSV success:", update_success)

# Test the takeSample method
time = 1.0  # Time of the state
dt = 0.1  # Time step
u_in_data = np.array([1.0, 2.0, 3.0])
result = generator.takeSample(u_in_data, time, dt)
print("takeSample result:", result)

# Call the writeSnapshot function
generator.writeSnapshot()

# Call the loadSamples function

# Create an instance of BasisGenerator
generator1 = libROM.BasisGenerator(options, incremental, basis_file_name,file_format)
result = generator1.takeSample(u_in_data, time, dt)

# Create a BasisWriter instance
base_file_name = "test_basisgenerator_file"
basis_writer = libROM.BasisWriter(generator1, base_file_name, file_format)
basis_writer.writeBasis("basis")

del basis_writer
del generator1 

kind = "basis"
cut_off = 10
db_format = libROM.Formats.HDF5
generator.loadSamples(base_file_name, kind, cut_off, db_format)

# Call the computeNextSampleTime function
u_in = [1.0, 2.0, 3.0]
rhs_in = [0.1, 0.2, 0.3]
time = 0.0
next_sample_time = generator.computeNextSampleTime(u_in, rhs_in, time)
print("computeNextSampleTime",next_sample_time)

# Get the spatial basis
spatial_basis = generator.getSpatialBasis()
print("spatial_basis",spatial_basis.getData())

# Get the temporal basis
temporal_basis = generator.getTemporalBasis()
print("temporal_basis",temporal_basis.getData())

# Get the singular values
singular_values = generator.getSingularValues()
print("singular_values",singular_values.getData())

# Get the snapshot matrix
snapshot_matrix = generator.getSnapshotMatrix()
print("snapshot_matrix",snapshot_matrix.getData())

# Get the number of basis time intervals
num_intervals = generator.getNumBasisTimeIntervals()
print("Number of basis time intervals:", num_intervals)

# Get the basis interval start time for a specific interval
interval = 0  # Replace with the desired interval index
interval_start_time = generator.getBasisIntervalStartTime(interval)
print("Start time for interval", interval, ":", interval_start_time)

# Get the number of samples taken
num_samples = generator.getNumSamples()
print("Number of samples:", num_samples)

# Test the endSamples method
generator.endSamples()

def test_plus():
    # Create an instance of BasisGenerator
    options = libROM.Options(4, 20, 3, True, True)
    incremental = False
    basis_file_name = "basis.h5"
    file_format = libROM.Formats.HDF5
    generator = libROM.BasisGenerator(options, incremental, basis_file_name, file_format)

    # Test the isNextSample method
    time = 1.0  
    is_next = generator.isNextSample(time)
    assert is_next

    # Test the updateRightSV method
    update_success = generator.updateRightSV()
    assert update_success 

    # Test the takeSample method
    time = 1.0  
    dt = 0.1  
    u_in_data = np.array([1.0, 2.0, 3.0])
    result = generator.takeSample(u_in_data, time, dt)
    assert result  

    # Call the writeSnapshot function
    generator.writeSnapshot()

    # Call the loadSamples function
    generator1 = libROM.BasisGenerator(options, incremental, basis_file_name, file_format)
    result = generator1.takeSample(u_in_data, time, dt)

    # Create a BasisWriter instance
    base_file_name = "test_basisgenerator_file"
    basis_writer = libROM.BasisWriter(generator1, base_file_name, file_format)
    basis_writer.writeBasis("basis")

    del basis_writer
    del generator1 

    kind = "basis"
    cut_off = 10
    db_format = libROM.Formats.HDF5
    generator.loadSamples(base_file_name, kind, cut_off, db_format)

    # Call the computeNextSampleTime function
    u_in = [1.0, 2.0, 3.0]
    rhs_in = [0.1, 0.2, 0.3]
    time = 0.0
    next_sample_time = generator.computeNextSampleTime(u_in, rhs_in, time)
    assert next_sample_time == 0.0

    # Get the spatial basis
    spatial_basis = generator.getSpatialBasis()
    assert spatial_basis.getData() == [[-0.2672612419124242, 0.9561828874675146], [-0.5345224838248487, -0.04390192218731964], [-0.8017837257372731, -0.2894596810309584], [-4.4e-323, -4.975416402579816e-309]]

    # Get the temporal basis
    temporal_basis = generator.getTemporalBasis()
    assert temporal_basis.getData() == [[-0.7071067811865476, 0.7071067811865474], [0.7071067811865474, 0.7071067811865476]]

    # Get the singular values
    singular_values = generator.getSingularValues()
    assert singular_values.getData() == [5.2915026221291805, 7.021666937153404e-16]

    # Get the snapshot matrix
    snapshot_matrix = generator.getSnapshotMatrix()
    assert snapshot_matrix.getData() == [[-3.7416573867739418, 3.74165738677394], [0.4217934441190679, 9.930136612989092e-16], [0.6326901661786019, 0.6180339887498948], [3.5e-323, 3.43792818009081e-309]]

    # Get the number of basis time intervals
    num_intervals = generator.getNumBasisTimeIntervals()
    assert num_intervals == 1

    # Get the basis interval start time for a specific interval
    interval = 0  # Replace with the desired interval index
    interval_start_time = generator.getBasisIntervalStartTime(interval)
    assert interval_start_time == 1.0

    # Get the number of samples taken
    num_samples = generator.getNumSamples()
    assert num_samples == 2

    # Test the endSamples method
    generator.endSamples()

if __name__ == "__main__":
    pytest.main()