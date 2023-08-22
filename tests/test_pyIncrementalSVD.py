import pytest
import sys
try:
    # import pip-installed package
    import pylibROM
    import pylibROM.linalg as libROM
    import pylibROM.linalg.svd as SVD 
except ModuleNotFoundError:
    # If pip-installed package is not found, import cmake-built package
    sys.path.append("../build")
    import _pylibROM as pylibROM
    import _pylibROM.linalg as libROM
    import _pylibROM.linalg.svd as SVD 
import numpy as np 

def test_getSpatialBasis():
    options = libROM.Options(3, 4)
    options.setMaxBasisDimension(3)
    options.setIncrementalSVD(1e-1, -1.0, -1.0, -1.0)
    incrementalSVD = SVD.IncrementalSVD(options, "irrelevant.txt" )
    assert incrementalSVD.getSpatialBasis() is None

def test_getTemporalBasis():
    options = libROM.Options(3, 4)
    options.setMaxBasisDimension(3)
    options.setIncrementalSVD(1e-1, -1.0, -1.0, -1.0)
    incrementalSVD = SVD.IncrementalSVD(options, "irrelevant.txt" )
    assert incrementalSVD.getTemporalBasis() is None

def test_getSingularValues():
    options = libROM.Options(3, 4)
    options.setMaxBasisDimension(3)
    options.setIncrementalSVD(1e-1, -1.0, -1.0, -1.0)
    incrementalSVD = SVD.IncrementalSVD(options, "irrelevant.txt" )
    assert incrementalSVD.getSingularValues() is None

def test_getSnapshotMatrix():
    options = libROM.Options(3, 4)
    options.setMaxBasisDimension(3)
    options.setIncrementalSVD(1e-1, -1.0, -1.0, -1.0)
    incrementalSVD = SVD.IncrementalSVD(options, "irrelevant.txt" )
    assert incrementalSVD.getSnapshotMatrix() is None

def test_getDim():
    options = libROM.Options(3, 4)
    options.setMaxBasisDimension(3)
    options.setIncrementalSVD(1e-1, -1.0, -1.0, -1.0)
    incrementalSVD = SVD.IncrementalSVD(options, "irrelevant.txt")
    assert incrementalSVD.getDim() == 3

def test_getNumBasisTimeIntervals():
    options = libROM.Options(3, 4)
    options.setMaxBasisDimension(3)
    options.setIncrementalSVD(1e-1, -1.0, -1.0, -1.0)
    incrementalSVD = SVD.IncrementalSVD(options, "irrelevant.txt" )
    num_intervals = incrementalSVD.getNumBasisTimeIntervals()
    assert num_intervals == 0

def test_getBasisIntervalStartTime():
    options = libROM.Options(3, 4)
    options.setMaxBasisDimension(3)
    options.setIncrementalSVD(1e-1, -1.0, -1.0, -1.0)
    incrementalSVD = SVD.IncrementalSVD(options, "irrelevant.txt" )
    num_intervals = incrementalSVD.getNumBasisTimeIntervals()
    for interval in range(num_intervals):
        start_time = incrementalSVD.getBasisIntervalStartTime(interval)
        assert interval == start_time

def test_isNewTimeInterval():
    options = libROM.Options(3, 4)
    options.setMaxBasisDimension(3)
    options.setIncrementalSVD(1e-1, -1.0, -1.0, -1.0)
    incrementalSVD = SVD.IncrementalSVD(options, "irrelevant.txt" )
    is_new_interval = incrementalSVD.isNewTimeInterval()
    assert is_new_interval == True

def test_increaseTimeInterval():
    options = libROM.Options(3, 4)
    options.setMaxBasisDimension(3)
    options.setIncrementalSVD(1e-1, -1.0, -1.0, -1.0)
    incrementalSVD = SVD.IncrementalSVD(options, "irrelevant.txt" )
    incrementalSVD.increaseTimeInterval()
    num_intervals = incrementalSVD.getNumBasisTimeIntervals()
    for interval in range(num_intervals):
        start_time = incrementalSVD.getBasisIntervalStartTime(interval)
        assert interval == start_time

def test_getNumSamples():
    options = libROM.Options(3, 4)
    options.setMaxBasisDimension(3)
    options.setIncrementalSVD(1e-1, -1.0, -1.0, -1.0)
    incrementalSVD = SVD.IncrementalSVD(options, "irrelevant.txt" )
    num_samples = incrementalSVD.getNumSamples()
    assert num_samples == 0

if __name__ == '__main__':
    pytest.main()



