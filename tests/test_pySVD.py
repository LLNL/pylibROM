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
    options = libROM.Options(4, 20, True, True)
    svd = SVD.SVD(options)
    dim = svd.getDim()
    assert dim == 4

def test_getMaxNumSamples():
    options = libROM.Options(4, 20, True, True)
    svd = SVD.SVD(options)
    assert(svd.getMaxNumSamples() == 20)

def test_getNumSamples():
    options = libROM.Options(4, 20, True, True)
    svd = SVD.SVD(options)
    num_samples = svd.getNumSamples()
    assert num_samples == 0

if __name__ == '__main__':
    pytest.main()

 




