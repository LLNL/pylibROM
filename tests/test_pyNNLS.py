#!/usr/bin/env python 
import pytest
import numpy as np
from scipy import optimize
from mpi4py import MPI
import _pylibROM.linalg as la

#- parallelization setup
comm = MPI.COMM_WORLD

#- input setup
A = np.random.rand(5,10)
x = np.zeros((10,))
rhs_lb, rhs_ub = np.ones((5,))*(3.5-1e-5), np.ones((5,))*(3.5+1e-5)
b = (rhs_lb+rhs_ub)/2
sol,res = optimize.nnls(A,b)


def test_getNumProcs():
    nnls = la.NNLSSolver()
    assert nnls.getNumProcs() == comm.size

def test_normalize_constraints():
    const_tol = 1e-6
    
    # normalize constraints in NNLS
    nnls = la.NNLSSolver(const_tol=const_tol)
    At_lr,x_lr = la.Matrix(A.T.copy(),True,True),la.Vector(x.copy(),True,True)
    lb_lr, ub_lr = la.Vector(rhs_lb.copy(),False,True), la.Vector(rhs_ub.copy(),False,True)
    nnls.normalize_constraints(At_lr,lb_lr,ub_lr)
    
    # normalize constraints manually
    halfgap_target = const_tol*1e3
    halfgap = (rhs_ub-rhs_lb)/2
    b = (rhs_lb + rhs_ub)/2
    An = np.zeros_like(A)
    rhs_lbn, rhs_ubn = np.zeros_like(rhs_lb), np.zeros_like(rhs_ub)
    for i in range(len(A)):
        s = halfgap_target/halfgap
        An[i] = A[i]*s[i]
        rhs_lbn, rhs_ubn = b*s - halfgap_target, b*s + halfgap_target

    assert np.allclose(At_lr.getData(),An.T,rtol=1e-10)
    assert np.allclose(lb_lr.getData(),rhs_lbn,rtol=1e-10)
    assert np.allclose(ub_lr.getData(),rhs_ubn,rtol=1e-10)


def test_solve_parallel_with_scalapack():
    nnls = la.NNLSSolver()
    At_lr,x_lr = la.Matrix(A.T.copy(),True,True),la.Vector(x.copy(),True,True)
    lb_lr, ub_lr = la.Vector(rhs_lb.copy(),False,True), la.Vector(rhs_ub.copy(),False,True)

    nnls.solve_parallel_with_scalapack(At_lr,lb_lr,ub_lr,x_lr)
    
    assert np.allclose(sol,x_lr.getData(),rtol=1e-10)

