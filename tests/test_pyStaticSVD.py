import sys
import pytest
sys.path.append("../build")

import pylibROM.linalg as libROM
import pylibROM.linalg.svd as SVD 
import numpy as np 

def test_getDim():
    options = libROM.Options(3, 10, 3, True, True)
    staticsvd=SVD.StaticSVD(options)
    dim = staticsvd.getDim()
    assert dim == 3

def test_getNumBasisTimeIntervals():
    options = libROM.Options(3, 10, 3, True, True)
    staticsvd=SVD.StaticSVD(options)
    num_intervals = staticsvd.getNumBasisTimeIntervals()
    assert num_intervals == 0

def test_getBasisIntervalStartTime():
    options = libROM.Options(3, 10, 3, True, True)
    staticsvd=SVD.StaticSVD(options)
    num_intervals = staticsvd.getNumBasisTimeIntervals()
    for interval in range(num_intervals):
        start_time = staticsvd.getBasisIntervalStartTime(interval)
        assert interval == start_time

def test_isNewTimeInterval():
    options = libROM.Options(3, 10, 3, True, True)
    staticsvd=SVD.StaticSVD(options)
    is_new_interval = staticsvd.isNewTimeInterval()
    assert is_new_interval == True

def test_increaseTimeInterval():
    options = libROM.Options(3, 10, 3, True, True)
    staticsvd=SVD.StaticSVD(options)
    staticsvd.increaseTimeInterval()
    num_intervals = staticsvd.getNumBasisTimeIntervals()
    for interval in range(num_intervals):
        start_time = staticsvd.getBasisIntervalStartTime(interval)
        assert interval == start_time

def test_getNumSamples():
    options = libROM.Options(4, 20, 3, True, True)
    staticsvd = SVD.StaticSVD(options)
    num_samples = staticsvd.getNumSamples()
    assert num_samples == 0

def test_takeSample():
    options = libROM.Options(3, 10, 3, True, True)
    staticsvd = SVD.StaticSVD(options)
    u_in = np.array([1.0, 2.0, 3.0, 4.0])
    time = 0.5
    add_without_increase = False
    result = staticsvd.takeSample(u_in, time, add_without_increase)
    assert result == True

def test_getSpatialBasis():
    options = libROM.Options(3, 10, 3, True, True)
    staticsvd = SVD.StaticSVD(options)
    u_in = np.array([1.0, 2.0, 3.0, 4.0])
    time = 0.5
    add_without_increase = False
    staticsvd.takeSample(u_in, time, add_without_increase)
    spatial_basis = staticsvd.getSpatialBasis()
    assert(np.array_equal(spatial_basis.getData(), [[-0.2672612419124243], [-0.5345224838248487], [-0.8017837257372731]]))

def test_getTemporalBasis():
    options = libROM.Options(3, 10, 3, True, True)
    staticsvd = SVD.StaticSVD(options)
    u_in = np.array([1.0, 2.0, 3.0, 4.0])
    time = 0.5
    add_without_increase = False
    staticsvd.takeSample(u_in, time, add_without_increase)
    temporal_basis = staticsvd.getTemporalBasis()
    assert(np.array_equal(temporal_basis.getData(), [[-1.0]]))

def test_getSingularValues():
    options = libROM.Options(3, 10, 3, True, True)
    staticsvd = SVD.StaticSVD(options)
    u_in = np.array([1.0, 2.0, 3.0, 4.0])
    time = 0.5
    add_without_increase = False
    staticsvd.takeSample(u_in, time, add_without_increase)
    singular_values = staticsvd.getSingularValues()
    assert(np.array_equal(singular_values.getData(), [3.7416573867739418]))

def test_getSnapshotMatrix():
    options = libROM.Options(5, 10, 5, True, True)
    staticsvd = SVD.StaticSVD(options)
    input_snapshot = np.array([[0.5377, 1.8339, -2.2588, 0.8622, 0.3188],
                           [-1.3077, -0.4336, 0.3426, 3.5784, 2.7694],
                           [-1.3499, 3.0349, 0.7254, -0.0631, 0.7147]])
    input_snapshot = input_snapshot.T
    time = 0.5
    add_without_increase = False
    for k in range(input_snapshot.shape[1]):
         result = staticsvd.takeSample(input_snapshot[:,k], time, add_without_increase)
    snapshot_matrix = staticsvd.getSnapshotMatrix()
    assert(np.array_equal(snapshot_matrix, input_snapshot)) 


if __name__ == '__main__':
    pytest.main()




