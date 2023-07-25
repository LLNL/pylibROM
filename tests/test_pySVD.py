import sys
import pytest
sys.path.append("..")

import build.pylibROM.linalg as libROM
import build.pylibROM.linalg.svd as SVD 
import numpy as np 

options = libROM.Options(4, 20, 3, True, True)
svd = SVD.SVD(options)

# Test the getDim() function
dim = svd.getDim()
print("Dimension:", dim)

# Test the getNumBasisTimeIntervals() function
num_intervals = svd.getNumBasisTimeIntervals()
print("Number of Basis Time Intervals:", num_intervals)

# Test the getBasisIntervalStartTime() function
for interval in range(num_intervals):
    start_time = svd.getBasisIntervalStartTime(interval)
    print(f"Start Time for Interval {interval}: {start_time}")

# Test the isNewTimeInterval() function
is_new_interval = svd.isNewTimeInterval()
print("Is New Time Interval:", is_new_interval)

# Test the increaseTimeInterval() function
svd.increaseTimeInterval()

# Test the getNumSamples() function
num_samples = svd.getNumSamples()
print("Number of Samples:", num_samples)

def test_plus():
    options = libROM.Options(4, 20, 3, True, True)
    svd = SVD.SVD(options)

    # Test the getDim() function
    dim = svd.getDim()
    assert dim == 4

    # Test the getNumBasisTimeIntervals() function
    num_intervals = svd.getNumBasisTimeIntervals()
    assert num_intervals == 0

    # Test the getBasisIntervalStartTime() function
    for interval in range(num_intervals):
        start_time = svd.getBasisIntervalStartTime(interval)
        assert interval == start_time

    # Test the isNewTimeInterval() function
    is_new_interval = svd.isNewTimeInterval()
    assert is_new_interval == True

    # Test the increaseTimeInterval() function
    svd.increaseTimeInterval()

    # Test the getNumSamples() function
    num_samples = svd.getNumSamples()
    assert num_samples == 0

if __name__ == '__main__':
    pytest.main()

 




