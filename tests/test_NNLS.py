#!/usr/bin/env python
import scipy as sp
import numpy as np
np.set_printoptions(precision=4)
from mpi4py import MPI
import sys
sys.path.append('..')
import build.pylibROM.linalg as la

#- parallelization setup
comm = MPI.COMM_WORLD
c_size,rank = comm.size, comm.rank

def parprint(*args):
    if rank == 0: print(*args)

def partition(N,comm_size):
    block_size, res = N//comm_size, N%comm_size
    block_sizes = [block_size+1]*res + [block_size]*(comm_size-res)
    block_indices = np.cumsum([0]+block_sizes)
    return block_indices, block_sizes

#- Problem setup & Matrix/vector dimensions
#n, m = int(sys.argv[1]), int(sys.argv[2]) # take size from inputs
n = int(sys.argv[1])
ms = list(map(int,sys.argv[2].strip('[').strip(']').split(',')))
# defining the numpy arrays
def sp_setup(n,m,rhs_avg,rhs_gap=0):
    np.random.seed(42)
    A,x = np.random.rand(n,m), np.zeros((m,))
    rhs_lb, rhs_ub = np.ones((n,))*(rhs_avg-rhs_gap/2), \
                     np.ones((n,))*(rhs_avg+rhs_gap/2)
    b, eps = (rhs_lb + rhs_ub)/2, (rhs_ub - rhs_lb)/2
    return A,x,b,eps,rhs_lb,rhs_ub
# defining libROM Matrix/Vectors from numpy arrays
# need to take care of data distribution among cores
def lr_setup(n,m,rhs_avg,rhs_gap=0,local_slice=np.s_[:]):
    A,x,b,eps,rhs_lb,rhs_ub = sp_setup(n,m,rhs_avg,rhs_gap)
    A_block, x_block = A[:,local_slice], x[local_slice]
    At_lr,x_lr = la.Matrix(A_block.T.copy(),True,True),la.Vector(x_block.copy(),True,True)
    lb_lr, ub_lr = la.Vector(rhs_lb,False,True), la.Vector(rhs_ub,False,True)
    return At_lr,x_lr,lb_lr,ub_lr

#- solving nnls with scipy (on single core)
comm.Barrier()
Ts_sp = []
if rank == 0:
    Sols_sp = []
    for m in ms:
        A,x,b,_,_,_ = sp_setup(n,m,rhs_avg=3.5,rhs_gap=0)
        ts = []
        for i in range(3):
            t1 = MPI.Wtime()
            sol,res = sp.optimize.nnls(A,b)
            t2 = MPI.Wtime()
            ts.append(t2-t1)
        Sols_sp.append(sol)
        Ts_sp.append(min(ts))
    #print(f'scipy result: sol = {sol}, with res = {res}')
#else:
#    sol = np.zeros((m,))
comm.Barrier()

#- solving nnls with librom (on multi-cores)
Ts_lr = []
Sols_lr = []
for m in ms:
    block_indices, block_sizes = partition(m,c_size)
    local_slice = np.s_[block_indices[rank]:block_indices[rank+1]]
    ts = []
    for i in range(3):
        At_lr,x_lr,lb_lr,ub_lr = lr_setup(n,m,rhs_avg=3.5,rhs_gap=0,local_slice=local_slice)
        nnls = la.NNLSSolver(verbosity=0)
        t1 = MPI.Wtime()
        nnls.solve_parallel_with_scalapack(At_lr,lb_lr,ub_lr,x_lr)
        t2 = MPI.Wtime()
        x_lr_gather = np.empty((m,))
        comm.Gatherv(np.array(x_lr.get_data()),(x_lr_gather,block_sizes),root=0)
        ts.append(t2-t1)
    Ts_lr.append(min(ts))
    Sols_lr.append(x_lr_gather)
parprint(f'For sizes m={ms}')
parprint(f'scipy execution times: {np.array(Ts_sp)} secs')
parprint(f'libROM execution times: {np.array(Ts_lr)} secs')

#- compare
if rank==0:
    print(f'Both producing identical results? \
           {[np.allclose(Sol_sp,Sol_lr) for Sol_sp,Sol_lr in zip(Sols_sp,Sols_lr)]}')
