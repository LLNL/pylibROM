import sys
import os.path as pth

sys.path.append(pth.join(pth.dirname(pth.abspath(__file__)), "../"))

import build.pylibROM.algo as algo
from build.pylibROM.algo import DMD
from build.pylibROM.linalg import Vector

import unittest
#from mpi4py import MPI


class GoogleTestFramework(unittest.TestCase):
    def test_GoogleTestFrameworkFound(self):
        self.assertTrue(True)

class DMDTest(unittest.TestCase):
    def test_DMD(self):
        mpi_init = MPI.Is_initialized()
        if mpi_init == False:
            MPI.Init()

        d_rank = MPI.COMM_WORLD.Get_rank()
        d_num_procs = MPI.COMM_WORLD.Get_size()

        num_total_rows = 5
        d_num_rows = num_total_rows // d_num_procs
        if num_total_rows % d_num_procs > d_rank:
            d_num_rows += 1

        row_offset = [0] * (d_num_procs + 1)
        row_offset[d_num_procs] = num_total_rows
        row_offset[d_rank] = d_num_rows

        row_offset = MPI.COMM_WORLD.allgather(row_offset)[0] #TODO: not sure if this works for multiple chls
        for i in range(d_num_procs - 1, -1, -1):
            row_offset[i] = row_offset[i + 1] - row_offset[i]

        sample1 = [0.5377, 1.8339, -2.2588, 0.8622, 0.3188]
        sample2 = [-1.3077, -0.4336, 0.3426, 3.5784, 2.7694]
        sample3 = [-1.3499, 3.0349, 0.7254, -0.0631, 0.7147]
        prediction_baseline = [-0.4344, -0.0974, 0.0542, 1.2544, 0.9610]

        dmd = DMD(d_num_rows, 1.0)
        dmd.takeSample(sample1[row_offset[d_rank]], 0.0)
        dmd.takeSample(sample2[row_offset[d_rank]], 1.0)
        dmd.takeSample(sample3[row_offset[d_rank]], 2.0)

        dmd.train(int(2))
        result = dmd.predict(3.0)

        for i in range(d_num_rows):
            self.assertAlmostEqual(result.item(i), prediction_baseline[row_offset[d_rank] + i], delta=1e-3)

        dmd.save("test_DMD")
        dmd_load = DMD.load("test_DMD")
        result_load = dmd_load.predict(3.0)

        for i in range(d_num_rows):
            self.assertAlmostEqual(result_load.item(i), prediction_baseline[row_offset[d_rank] + i], delta=1e-3)

#Not right now...
#if __name__ == "__main__":
#    unittest.main()
"""
mpi_init = MPI.Is_initialized()
if mpi_init == False:
    MPI.Init()

d_rank = MPI.COMM_WORLD.Get_rank()
d_num_procs = MPI.COMM_WORLD.Get_size()

num_total_rows = 5
d_num_rows = num_total_rows // d_num_procs
if num_total_rows % d_num_procs > d_rank:
    d_num_rows += 1


row_offset = [0] * (d_num_procs + 1)
row_offset[d_num_procs] = num_total_rows
row_offset[d_rank] = d_num_rows

row_offset = MPI.COMM_WORLD.allgather(row_offset)[0]

for i in range(d_num_procs - 1, -1, -1):
    row_offset[i] = row_offset[i + 1] - row_offset[i]

"""
d_num_rows = 5

sample1 = [0.5377, 1.8339, -2.2588, 0.8622, 0.3188]
sample2 = [-1.3077, -0.4336, 0.3426, 3.5784, 2.7694]
sample3 = [-1.3499, 3.0349, 0.7254, -0.0631, 0.7147]
prediction_baseline = [-0.4344, -0.0974, 0.0542, 1.2544, 0.9610]

dmd = DMD(d_num_rows, 1.0)

print("was able to instantiate DMD")
#dmd.takeSample(sample1[row_offset[d_rank]], 0.0)
#dmd.takeSample(sample2[row_offset[d_rank]], 1.0)
#dmd.takeSample(sample3[row_offset[d_rank]], 2.0)
dmd.takeSample(sample1[1], 0.0)
dmd.takeSample(sample2[2], 1.0)
dmd.takeSample(sample3[3], 2.0)


print("was able to take sample")
#Actually struggling to provide an int... dmd.train(int(2))   #note that you have to be explicit with data types
dmd.train(1)
print("Was able to train")
"""

csv_path = pth.join(pth.dirname(pth.abspath(__file__)), "dmd_data/dmd_csv/dmd_data/dmd_par1/step0/sol.csv")
hf5_path = pth.join(pth.dirname(pth.abspath(__file__)),"../extern/libROM/examples/dmd/hc_par0.log")


print(f'will load {hf5_path}')
testDMD = algo.DMD(hf5_path) #TODO what is a good golden file for this?? not clear...

print("loadedTestDMD")


# Create the necessary inputs
dim = 3
dt = 0.1
alt_output_basis = True
state_offset = None



#Instantiate DMD to check if anything works at all...
#testDMD = algo.DMD(dim, dt, alt_output_basis, state_offset)

#Instantiate a DMD instance with real data, so predictions should be... predictable
"""
