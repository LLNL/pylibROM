import pytest
import numpy as np
import sys
try:
    # import pip-installed package
    import pylibROM
    import pylibROM.linalg as linalg 
    import pylibROM.algo as algo 
    import pylibROM.utils as utils
except ModuleNotFoundError:
    # If pip-installed package is not found, import cmake-built package
    sys.path.append("../build")
    import _pylibROM as pylibROM
    import _pylibROM.linalg as linalg
    import _pylibROM.algo as algo
    import _pylibROM.utils as utils

def test_DMD():
    from mpi4py import MPI
    d_rank = MPI.COMM_WORLD.Get_rank()
    d_num_procs = MPI.COMM_WORLD.Get_size()

    num_total_rows = 5
    d_num_rows = utils.split_dimension(num_total_rows, MPI.COMM_WORLD)
    num_total_rows_check, row_offset = utils.get_global_offsets(d_num_rows, MPI.COMM_WORLD)
    assert(num_total_rows == num_total_rows_check)

    samples = [[0.5377, 1.8339, -2.2588, 0.8622, 0.3188],
                [-1.3077, -0.4336, 0.3426, 3.5784, 2.7694],
                [-1.3499, 3.0349, 0.7254, -0.0631, 0.7147]]
    prediction_baseline = [-0.4344, -0.0974, 0.0542, 1.2544, 0.9610]

    dmd = algo.DMD(d_num_rows, 1.0)
    for k, sample in enumerate(samples):
        dmd.takeSample(sample[row_offset[d_rank]:row_offset[d_rank]+d_num_rows], k * 1.0)

    dmd.train(2)
    result = dmd.predict(3.0)
    # print("rank: %d, " % d_rank, result.getData())
    assert np.allclose(result.getData(), prediction_baseline[row_offset[d_rank]:row_offset[d_rank]+d_num_rows], atol=1e-3)

    dmd.save("test_DMD")
    dmd_load = algo.DMD("test_DMD")
    result_load = dmd_load.predict(3.0)

    assert np.allclose(result_load.getData(), prediction_baseline[row_offset[d_rank]:row_offset[d_rank]+d_num_rows], atol=1e-3)

if __name__ == '__main__':
    pytest.main()
