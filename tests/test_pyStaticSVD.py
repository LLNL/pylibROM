import sys
import pytest
sys.path.append("../build")

import pylibROM.linalg as libROM
import pylibROM.linalg.svd as SVD 
import numpy as np 

options = libROM.Options(3, 10, 3, True, True)
staticsvd = SVD.StaticSVD(options)

# Test the getDim() function
dim = staticsvd.getDim()
print("getDim():", dim)

# Test the getNumBasisTimeIntervals() function
num_intervals = staticsvd.getNumBasisTimeIntervals()
print("getNumBasisTimeIntervals():", num_intervals)

# Test the getBasisIntervalStartTime() function
for interval in range(num_intervals):
    start_time = staticsvd.getBasisIntervalStartTime(interval)
    print(f"getBasisIntervalStartTime({interval}):", start_time)

# Test the isNewTimeInterval() function
is_new_interval = staticsvd.isNewTimeInterval()
print("isNewTimeInterval():", is_new_interval)

# Test the increaseTimeInterval() function
staticsvd.increaseTimeInterval()

# Test the getNumSamples() function
num_samples = staticsvd.getNumSamples()
print("getNumSamples():", num_samples)


# Test the takeSample function
u_in = np.array([1.0, 2.0, 3.0, 4.0])
time = 0.5
add_without_increase = False
result = staticsvd.takeSample(u_in, time, add_without_increase)
print("takeSample() result:", result)
    
# Test the getSpatialBasis function
spatial_basis = staticsvd.getSpatialBasis()
print("getSpatialBasis():", spatial_basis.getData())

# Test the getTemporalBasis function
temporal_basis = staticsvd.getTemporalBasis()
print("getTemporalBasis():", temporal_basis.getData())

# Test the getSingularValues function
singular_values = staticsvd.getSingularValues()
print("getSingularValues():", singular_values.getData())

# Test the getSnapshotMatrix function
snapshot_matrix = staticsvd.getSnapshotMatrix()
print("getSnapshotMatrix():", snapshot_matrix.getData())


def test_plus():
    options = libROM.Options(3, 10, 3, True, True)
    staticsvd=SVD.StaticSVD(options)

    # Test the getDim() function
    dim = staticsvd.getDim()
    assert dim == 3

    # Test the getNumBasisTimeIntervals() function
    num_intervals = staticsvd.getNumBasisTimeIntervals()
    assert num_intervals == 0

    # Test the getBasisIntervalStartTime() function
    for interval in range(num_intervals):
        start_time = staticsvd.getBasisIntervalStartTime(interval)
        assert interval == start_time

    # Test the isNewTimeInterval() function
    is_new_interval = staticsvd.isNewTimeInterval()
    assert is_new_interval == True

    # Test the increaseTimeInterval() function
    staticsvd.increaseTimeInterval()

    # Test the getNumSamples() function
    num_samples = staticsvd.getNumSamples()
    assert num_samples == 0

    # Test the takeSample function
    u_in = np.array([1.0, 2.0, 3.0, 4.0])
    time = 0.5
    add_without_increase = False
    result = staticsvd.takeSample(u_in, time, add_without_increase)
    assert result == True
   
    # Test the getSpatialBasis function
    spatial_basis = staticsvd.getSpatialBasis()
    assert spatial_basis.getData() == [[-0.26726124],[-0.53452248],[-0.80178373]]

    # Test the getTemporalBasis function
    temporal_basis = staticsvd.getTemporalBasis()
    assert temporal_basis.getData() == [[-1.0]]

    # Test the getSingularValues function
    singular_values = staticsvd.getSingularValues()
    assert singular_values.getData() == [3.74165739]

    # Test the getSnapshotMatrix function
    snapshot_matrix = staticsvd.getSnapshotMatrix()
    assert snapshot_matrix.getData() == [[-3.74165739],[ 0.42179344],[ 0.63269017]]


if __name__ == '__main__':
    pytest.main()


