# import sys
# sys.path.append("..")

import pylibROM.linalg as libROM
import numpy as np 

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
# base_file_name = "basis.h5_snapshot"
# kind = "snapshot"
# cut_off = 10
# db_format = libROM.Formats.HDF5
# generator.loadSamples(base_file_name, kind, cut_off, db_format)

# Call the computeNextSampleTime function
u_in = [1.0, 2.0, 3.0]
rhs_in = [0.1, 0.2, 0.3]
time = 0.0
next_sample_time = generator.computeNextSampleTime(u_in, rhs_in, time)
print("computeNextSampleTime",next_sample_time)

# Get the spatial basis
spatial_basis = generator.getSpatialBasis()
print("spatial_basis",spatial_basis.get_data())

# Get the temporal basis
temporal_basis = generator.getTemporalBasis()
print("temporal_basis",temporal_basis.get_data())

# Get the singular values
singular_values = generator.getSingularValues()
print("singular_values",singular_values.get_data())

# Get the snapshot matrix
snapshot_matrix = generator.getSnapshotMatrix()
print("snapshot_matrix",snapshot_matrix.get_data())

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
