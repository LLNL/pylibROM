import sys
import pytest
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

def test_getDim():
    options = libROM.Options(4, 20, 3, True, True)
    svd = SVD.SVD(options)
    dim = svd.getDim()
    assert dim == 4

def test_getNumBasisTimeIntervals():
    options = libROM.Options(4, 20, 3, True, True)
    svd = SVD.SVD(options)
    num_intervals = svd.getNumBasisTimeIntervals()
    assert num_intervals == 0

def test_getBasisIntervalStartTime():
    options = libROM.Options(4, 20, 3, True, True)
    svd = SVD.SVD(options)
    num_intervals = svd.getNumBasisTimeIntervals()
    for interval in range(num_intervals):
        start_time = svd.getBasisIntervalStartTime(interval)
        assert interval == start_time

def test_isNewTimeInterval():
    options = libROM.Options(4, 20, 3, True, True)
    svd = SVD.SVD(options)
    is_new_interval = svd.isNewTimeInterval()
    assert is_new_interval == True

def test_increaseTimeInterval():
    options = libROM.Options(4, 20, 3, True, True)
    svd = SVD.SVD(options)
    svd.increaseTimeInterval()
    num_intervals = svd.getNumBasisTimeIntervals()
    for interval in range(num_intervals):
        start_time = svd.getBasisIntervalStartTime(interval)
        assert interval == start_time

def test_getNumSamples():
    options = libROM.Options(4, 20, 3, True, True)
    svd = SVD.SVD(options)
    num_samples = svd.getNumSamples()
    assert num_samples == 0

if __name__ == '__main__':
    pytest.main()

 




