import sys
import pytest
sys.path.append("../build")
import _pylibROM.linalg as linalg
import numpy as np 
import _pylibROM.hyperreduction as hyperreduction


@pytest.mark.skip("TODO")
def test_SpaceTimeSampling():
        # Prepare test data
        num_f_basis_vectors_used = 5
        num_cols = 5
        num_rows = 10
        num_t_samples_req=5
        num_s_samples_req= 10
        s_basis = linalg.Matrix(num_rows,num_cols,True,False)
        t_basis = linalg.Matrix(num_rows,num_cols,False,False)
        f_sampled_row = np.zeros(num_s_samples_req, dtype=np.int32)
        f_sampled_rows_per_proc = np.zeros(1, dtype=np.int32)
        s_basis_sampled = linalg.Matrix(num_s_samples_req,num_t_samples_req,False,False)
        myid = 0
        num_procs = 1

        t_samples= hyperreduction.SpaceTimeSampling(s_basis, t_basis, num_f_basis_vectors_used,
                                                    f_sampled_row,f_sampled_rows_per_proc, s_basis_sampled,
                                                    myid, num_procs,num_t_samples_req,num_s_samples_req,False)
        assert np.all(t_samples == [0, 1, 2, 3, 4])
        assert np.all(f_sampled_row == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        assert np.all(f_sampled_rows_per_proc == [10])


@pytest.mark.skip("TODO")
def test_GetSampledSpaceTimeBasis():
        t_samples = [1,2,3]
        t_basis = linalg.Matrix(3,5,True,False)
        t_basis.fill(2.0)
        s_basis_sampled = linalg.Matrix(2,5,False,False)
        s_basis_sampled.fill(3.0)
        f_basis_sampled_inv = linalg.Matrix(6, 2, False,False)
        hyperreduction.GetSampledSpaceTimeBasis(t_samples, t_basis, s_basis_sampled, f_basis_sampled_inv)
        assert np.allclose(f_basis_sampled_inv.getData(),[[1728.0, 6.0], [1728.0, 6.0], [4.751093e-318, 3.297888e-320], [1728.0, 6.0], [1728.0, 6.0], [4.751093e-318, 3.297888e-320]])


if __name__ == '__main__':
    pytest.main()

