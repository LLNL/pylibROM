# import sys
# sys.path.append("..")

import pylibROM.linalg as libROM
import numpy as np 

#Testing with two arguments
options = libROM.Options(4, 20)
print("dim",options.dim)
print("max_basis_dimension",options.max_basis_dimension)
print("max_num_samples",options.max_num_samples)
print("update_right_SV",options.update_right_SV)
print("write_snapshots",options.write_snapshots)

#Testing with all arguments
options = libROM.Options(4, 20,True,False)
print("dim",options.dim)
print("max_num_samples",options.max_num_samples)
print("update_right_SV",options.update_right_SV)
print("write_snapshots",options.write_snapshots)

# Test the member functions
print("set max_basis_dimension value to 10 using setMaxBasisDimension",options.setMaxBasisDimension(10).max_basis_dimension)
print("set singular_value_tol value to 0.01 using setSingularValueTol",options.setSingularValueTol(0.01).singular_value_tol)
print("set debug_algorithm to true using setDebugMode",options.setDebugMode(True).debug_algorithm)
options.setRandomizedSVD(True, randomized_subspace_dim_=5, random_seed_=42)
options.setIncrementalSVD(linearity_tol_=0.001, initial_dt_=0.1, sampling_tol_=0.001, max_time_between_samples_=1.0, fast_update_=True, skip_linearly_dependent_=False)
options.setStateIO(save_state_=True, restore_state_=False)
options.setSamplingTimeStepScale(min_sampling_time_step_scale_=0.5, sampling_time_step_scale_=1.0, max_sampling_time_step_scale_=2.0)


print("set randomized to true using setRandomizedSVD",options.randomized)
print("set randomized_subspace_dim to 5 using setRandomizedSVD",options.randomized_subspace_dim)
print("set random_seed value to 42 using setRandomizedSVD",options.random_seed)
print("set linearity_tol value to 0.001 using setIncrementalSVD",options.linearity_tol)
print("set initial_dt value to 0.1 using setIncrementalSVD",options.initial_dt)
print("set sampling_tol value to 0.001 using setIncrementalSVD",options.sampling_tol)
print("set max_time_between_samples value to 1.0 using setIncrementalSVD",options.max_time_between_samples)
print("set fast_update value to true using setIncrementalSVD",options.fast_update)
print("set skip_linearly_dependent value to false using setIncrementalSVD",options.skip_linearly_dependent)
print("set save_state value to true using setStateIO",options.save_state)
print("set restore_state value to false using setStateIO",options.restore_state)
print("set min_sampling_time_step_scale value to 0.5 using setSamplingTimeStepScale",options.min_sampling_time_step_scale)
print("set sampling_time_step_scale value to 1.0 using setSamplingTimeStepScale",options.sampling_time_step_scale)
print("set max_sampling_time_step_scale value to 2.0 using setSamplingTimeStepScale",options.max_sampling_time_step_scale)