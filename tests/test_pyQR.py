import sys
import pytest
import mpi4py
import numpy as np
try:
    # import pip-installed package
    import pylibROM
    import pylibROM.linalg as libROM
except ModuleNotFoundError:
    # If pip-installed package is not found, import cmake-built package
    sys.path.append("../build")
    import _pylibROM as pylibROM
    import _pylibROM.linalg as libROM


def test_MatrixQR():
    assert mpi4py.MPI.Is_initialized()
    comm = mpi4py.MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_procs = comm.Get_size()

    num_total_rows = 5
    num_columns = 3
    loc_num_rows = pylibROM.split_dimension(num_total_rows, comm)
    total_rows, row_offset = pylibROM.get_global_offsets(loc_num_rows, comm)
    assert total_rows == num_total_rows

    q_data = [
        3.08158946098238906153E-01,    -9.49897947980619661301E-02,    -4.50691774108525788911E-01,
        -1.43697905723455976457E-01,    9.53289043424090820622E-01,    8.77767692937209131898E-02,
        -2.23655845793717528158E-02,    -2.10628953513210204207E-01,   8.42235962392685943989E-01,
        -7.29903965154318323805E-01,    -1.90917141788945754488E-01,   -2.77280930877637610266E-01,
        -5.92561353877168350834E-01,    -3.74570084880578441089E-02,   5.40928141934190823137E-02
    ]

    r_data = [
        -1.78651649346571794741E-01,     5.44387957786310106023E-01,    -8.19588518467042281834E-01,
        0.0,                             -3.13100149275943651084E-01,   -9.50441422536040881122E-04,
        0.0,                             0.0,                            5.72951792961765460355E-01
    ]

    exactQ = libROM.Matrix(loc_num_rows, num_columns, True)
    exactR = libROM.Matrix(np.asarray(r_data).reshape((3,3)), False, True)

    for i in range(loc_num_rows):
        for j in range(num_columns):
            exactQ[i,j] = q_data[((i + row_offset[rank]) * num_columns) + j]

    assert exactQ.numRows() == loc_num_rows
    assert exactQ.numColumns() == num_columns

    # Verify that the columns of Q are orthonormal
    id = exactQ.transposeMult(exactQ)
    assert id.numRows() == num_columns
    assert id.numColumns() == num_columns

    maxError = np.max(np.abs(id.getData() - np.eye(num_columns)))
    np.testing.assert_almost_equal(maxError, 0.0, 15)


    # Compute A = QR
    # A = exactQ.mult(exactR) # need PR28 fix for this syntax
    A = libROM.Matrix(num_columns, num_columns, True)
    exactQ.mult(exactR, A)

    # Factorize A
    QRfactors = A.qr_factorize()
    assert len(QRfactors) == 2

    Q = QRfactors[0]
    R = QRfactors[1]
    assert Q.numRows() == loc_num_rows
    assert Q.numColumns() == num_columns
    assert R.numRows() == num_columns
    assert R.numColumns() == num_columns

    # Verify that Q == -exactQ and R == -exactR
    maxError = np.max(np.abs(exactQ.getData() + Q.getData()))
    np.testing.assert_almost_equal(maxError, 0.0, 15)

    maxError = np.max(np.abs(exactR.getData() + R.getData()))
    np.testing.assert_almost_equal(maxError, 0.0, 15)


if __name__ == '__main__':
    pytest.main()
