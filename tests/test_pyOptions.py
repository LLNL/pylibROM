import sys
import pytest
try:
    # import pip-installed package
    import pylibROM
    import pylibROM.linalg as libROM
except ModuleNotFoundError:
    # If pip-installed package is not found, import cmake-built package
    sys.path.append("../build")
    import _pylibROM as pylibROM
    import _pylibROM.linalg as libROM
import numpy as np 

def test_options_constructor_two_args():
    options = libROM.Options(4, 20)
    assert options.dim == 4
    assert options.max_basis_dimension == 20
    assert options.max_num_samples == 20
    assert not options.update_right_SV 
    assert not options.write_snapshots 

def test_options_constructor_all_args():
    options = libROM.Options(4, 20, True, False)
    assert options.dim == 4
    assert options.max_basis_dimension == 20
    assert options.max_num_samples == 20
    assert options.update_right_SV
    assert not options.write_snapshots

def test_setMaxBasisDimension():
    options = libROM.Options(4, 5, True, False)
    options.setMaxBasisDimension(10)
    assert options.max_basis_dimension == 10

def test_setSingularValueTolerance():
    options = libROM.Options(4, 20, True, False)
    options.setSingularValueTol(0.01)
    assert options.singular_value_tol == 0.01

def test_setDebugMode():
    options = libROM.Options(4, 20, True, False)
    options.setDebugMode(True)
    assert options.debug_algorithm

def test_setRandomizedSVD():
    options = libROM.Options(4, 20, True, False)
    options.setRandomizedSVD(True, randomized_subspace_dim_=5, random_seed_=42)
    assert options.randomized
    assert options.randomized_subspace_dim == 5
    assert options.random_seed == 42

def test_setIncrementalSVD():
    options = libROM.Options(4, 20, True, False)
    options.setIncrementalSVD(linearity_tol_=0.001, initial_dt_=0.1, sampling_tol_=0.001, max_time_between_samples_=1.0, fast_update_=True, skip_linearly_dependent_=False)
    assert options.linearity_tol == 0.001
    assert options.initial_dt == 0.1
    assert options.sampling_tol == 0.001
    assert options.max_time_between_samples == 1.0
    assert options.fast_update
    assert not options.skip_linearly_dependent

def test_setStateIO():
    options = libROM.Options(4, 20, True, False)
    options.setStateIO(save_state_=True, restore_state_=False)
    assert options.save_state
    assert not options.restore_state

def test_setSamplingTimeStepScale():
    options = libROM.Options(4, 20, True, False)
    options.setSamplingTimeStepScale(min_sampling_time_step_scale_=0.5, sampling_time_step_scale_=1.0, max_sampling_time_step_scale_=2.0)
    assert options.min_sampling_time_step_scale == 0.5
    assert options.sampling_time_step_scale == 1.0
    assert options.max_sampling_time_step_scale == 2.0


if __name__ == '__main__':
    pytest.main()