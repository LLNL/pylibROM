import pytest
import sys
sys.path.append("..")

import build.pylibROM.linalg as libROM
import build.pylibROM.linalg.svd as SVD 
import numpy as np 


class FakeIncrementalSVD(SVD.IncrementalSVD):
    def __init__(self, options, basis_file_name):
        super().__init__(options, basis_file_name)
        dim = options.dim

        # Construct a fake d_U, d_S, d_basis
        self.d_basis = libROM.Matrix(dim,dim, False,False)
        self.d_S = libROM.Vector(dim, False)

        # Use the identity matrix as a fake basis and fake singular values
        for i in range(dim):
            for j in range(i):
                self.d_basis.__setitem__(i, j,0) 
                self.d_basis.__setitem__(j, i,0) 
            self.d_basis.__setitem__(i, i,1) 
            self.d_S.__setitem__(i,1)


options = libROM.Options(3, 4)
options.setMaxBasisDimension(3)
options.setIncrementalSVD(1e-1, -1.0, -1.0, -1.0)
incrementalSVD = FakeIncrementalSVD(options, "irrelevant.txt" )
# B = incrementalSVD.getSpatialBasis()
# for i in range(incrementalSVD.getDim()):
#         for j in range(i):
#                 assert B(i, j) == 0
#                 assert B(j, i) == 0
#         assert B(i, i) == 1
# S = incrementalSVD.getSingularValues()
# for i in range(incrementalSVD.getDim()):
#             assert S(i) == 1

def test_plus():
    incrementalSVD1 = SVD.IncrementalSVD(options, "irrelevant.txt" )
    assert incrementalSVD1.getDim() == 3

if __name__ == '__main__':
    pytest.main()



