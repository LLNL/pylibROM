# //               libROM MFEM Example: parametric ROM for nonlinear elasticity problem (adapted from ex10p.cpp)
# //
# // Compile with: make nonlinear_elasticity_global_rom
# //
# // Description:  This examples solves a time dependent nonlinear elasticity
# //               problem of the form dv/dt = H(x) + S v, dx/dt = v, where H is a
# //               hyperelastic model and S is a viscosity operator of Laplacian
# //               type. The geometry of the domain is assumed to be as follows:
# //
# //                                 +---------------------+
# //                    boundary --->|                     |
# //                    attribute 1  |                     |
# //                    (fixed)      +---------------------+
# //
# //               The example demonstrates the use of hyper reduction to solve a
# //               nonlinear elasticity problem. Time integration is done with various
# //               explicit time integrator solvers. The basis for the velocity field
# //               is either constructed using a separate velocity basis or using the
# //               displacement basis. It is possible to set the initial condition in
# //               terms of either velocity or deformation. The velocity initial condition
# //               works better when both velocity and displacement bases are used. The
# //               deformation initial condition is better when only the displacement
# //               basis is used. The input flag -sc controls the scaling of the initial
# //               condition applied. This is what parameterizes the ROM. If the scaling
# //               factor is within the range +-10%, the results are generally accurate.

# // =================================================================================
# //
# // Sample runs and results for parametric ROM using displacement basis, velocity basis
# // and nonlinear term basis, with velocity initial condition:
# //
# // Offline phase:
# //      python3 nonlinear_elasticity_global_rom.py -offline -dt 0.01 -tf 5.0 -s 14 -vs 100 -sc 0.90 -id 0
# //
# //      python3 nonlinear_elasticity_global_rom.py -offline -dt 0.01 -tf 5.0 -s 14 -vs 100 -sc 1.10 -id 1
# //
# // Merge phase:
# //      python3 nonlinear_elasticity_global_rom.py -merge -ns 2 -dt 0.01 -tf 5.0
# //
# // Create FOM comparison data:
# //      python3 nonlinear_elasticity_global_rom.py -offline -dt 0.01 -tf 5.0 -s 14 -vs 100 -sc 1.00 -id 2
# //
# // Online phase with full sampling:
# //      python3 nonlinear_elasticity_global_rom.py -online -dt 0.01 -tf 5.0 -s 14 -vs 100 -hyp -rvdim 40 -rxdim 10 -hdim 71 -nsr 1170 -sc 1.00
# // Output message:
#         Elapsed time for time integration loop 5.87029e+00
#         Relative error of ROM position (x) at t_final: 5.000000 is 2.31698489e-04
#         Relative error of ROM velocity (v) at t_final: 5.000000 is 4.66941161e-01
#         Elapsed time for entire simulation 9.86927e+00
# //
# // =================================================================================
# //
# // Sample runs and results for parametric ROM using displacement basis, velocity basis
# // and nonlinear term basis, with velocity initial condition: (3d case in librom.net)
# //
# // Offline phase:
# //      python3 nonlinear_elasticity_global_rom.py --mesh ../data/beam-hex-nurbs.mesh -offline -dt 0.01 -tf 5.0 -s 14 -vs 10 -sc 3.90 -id 0
# //
# //      python3 nonlinear_elasticity_global_rom.py --mesh ../data/beam-hex-nurbs.mesh -offline -dt 0.01 -tf 5.0 -s 14 -vs 10 -sc 4.10 -id 1
# //
# // Merge phase:
# //      python3 nonlinear_elasticity_global_rom.py --mesh ../data/beam-hex-nurbs.mesh -merge -ns 2 -dt 0.01 -tf 5.0
# //
# // Create FOM comparison data:
# //      python3 nonlinear_elasticity_global_rom.py --mesh ../data/beam-hex-nurbs.mesh -offline -dt 0.01 -tf 5.0 -s 14 -vs 5 -sc 3.92 -id 2
# //
# // Online phase with full sampling:
# //      python3 nonlinear_elasticity_global_rom.py --mesh ../data/beam-hex-nurbs.mesh -online -dt 0.01 -tf 5.0 -s 14 -vs 5 -hyp -rvdim 40 -rxdim 10 -hdim 71 -nsr 200 -sc 3.92
# // Output message:
#         Elapsed time for time integration loop 3.47888e+01
#         Relative error of ROM position (x) at t_final: 5.000000 is 5.08981700e-03
#         Relative error of ROM velocity (v) at t_final: 5.000000 is 9.11743695e+00
#         Elapsed time for entire simulation 6.63521e+01
# //
# // =================================================================================
# //
# // This example runs in parallel with MPI, by using the same number of MPI ranks
# // in all phases (offline, merge, online).

import os
import io
import sys
import time
try:
    import mfem.par as mfem
except ModuleNotFoundError:
    msg = "PyMFEM is not installed yet. Install PyMFEM:\n"
    msg += "\tgit clone https://github.com/mfem/PyMFEM.git\n"
    msg += "\tcd PyMFEM\n"
    msg += "\tpython3 setup.py install --with-parallel\n"
    raise ModuleNotFoundError(msg)

from ctypes import c_double
from mfem.par import intArray, add_vector, subtract_vector
from os.path import expanduser, join, dirname
import numpy as np
from numpy import sin, cos, exp, sqrt, pi, abs, array, floor, log, sum
from scipy.special import erfc

sys.path.append("../../build")
import pylibROM.linalg as linalg
import pylibROM.hyperreduction as hyper
import pylibROM.mfem as mfem_support
from pylibROM.mfem import ComputeCtAB
from pylibROM.python_utils import StopWatch

class HyperelasticOperator(mfem.PyTimeDependentOperator):
    def __init__(self, fespace, ess_tdof_list_, visc, mu, K):
        mfem.PyTimeDependentOperator.__init__(self, 2*fespace.TrueVSize(), 0.0)

        rel_tol = 1e-8
        skip_zero_entries = 0
        ref_density = 1.0

        self.ess_tdof_list = ess_tdof_list_
        self.z = mfem.Vector(self.Height() // 2)
        self.z2 = mfem.Vector(self.Height() // 2)
        self.H_sp = mfem.Vector(self.Height() // 2)
        self.dvxdt_sp = mfem.Vector(self.Height() // 2)
        self.fespace = fespace
        self.viscosity = visc

        M = mfem.ParBilinearForm(fespace)
        S = mfem.ParBilinearForm(fespace)
        H = mfem.ParNonlinearForm(fespace)
        self.M = M
        self.H = H
        self.S = S

        rho = mfem.ConstantCoefficient(ref_density)
        M.AddDomainIntegrator(mfem.VectorMassIntegrator(rho))
        M.Assemble(skip_zero_entries)
        M.Finalize(skip_zero_entries)
        self.Mmat = M.ParallelAssemble()
        self.Mmat.EliminateRowsCols(self.ess_tdof_list)

        M_solver = mfem.CGSolver(fespace.GetComm())
        M_prec = mfem.HypreSmoother()
        M_solver.iterative_mode = False
        M_solver.SetRelTol(rel_tol)
        M_solver.SetAbsTol(0.0)
        M_solver.SetMaxIter(30)
        M_solver.SetPrintLevel(0)
        M_prec.SetType(mfem.HypreSmoother.Jacobi)
        M_solver.SetPreconditioner(M_prec)
        M_solver.SetOperator(self.Mmat)

        self.M_solver = M_solver
        self.M_prec = M_prec

        model = mfem.NeoHookeanModel(mu, K)
        H.AddDomainIntegrator(mfem.HyperelasticNLFIntegrator(model))
        H.SetEssentialTrueDofs(self.ess_tdof_list)
        self.model = model

        visc_coeff = mfem.ConstantCoefficient(visc)
        S.AddDomainIntegrator(mfem.VectorDiffusionIntegrator(visc_coeff))
        S.Assemble(skip_zero_entries)
        S.Finalize(skip_zero_entries)
        self.Smat = mfem.HypreParMatrix()
        S.FormSystemMatrix(self.ess_tdof_list, self.Smat)

    def Mult(self, vx, dvx_dt):
        sc = self.Height() // 2
        v = mfem.Vector(vx, 0,  sc)
        x = mfem.Vector(vx, sc,  sc)
        dv_dt = mfem.Vector(dvx_dt, 0, sc)
        dx_dt = mfem.Vector(dvx_dt, sc,  sc)

        self.H.Mult(x, self.z)
        self.H_sp.Assign(self.z)

        if (self.viscosity != 0.0):
            self.Smat.Mult(v, self.z2)
            self.z += self.z2
        
        self.z.Neg()
        self.M_solver.Mult(self.z, dv_dt)

        dx_dt.Assign(v) # this changes dvx_dt
        self.dvxdt_sp.Assign(dvx_dt)

    def ElasticEnergy(self, x):
        return self.H.GetEnergy(x)

    def KineticEnergy(self, v):
        from mpi4py import MPI
        local_energy = 0.5*self.M.InnerProduct(v, v)
        energy = MPI.COMM_WORLD.allreduce(local_energy, op=MPI.SUM)
        return energy

    def GetElasticEnergyDensity(self, x, w):
        w_coeff = ElasticEnergyCoefficient(self.model, x)
        w.ProjectCoefficient(w_coeff)

class RomOperator(mfem.PyTimeDependentOperator):
    # RomOperator::RomOperator(HyperelasticOperator* fom_,
    #                      HyperelasticOperator* fomSp_, const int rvdim_, const int rxdim_,
    #                      const int hdim_, CAROM::SampleMeshManager* smm_, const Vector* v0_,
    #                      const Vector* x0_, const Vector v0_fom_, const CAROM::Matrix* V_v_,
    #                      const CAROM::Matrix* V_x_, const CAROM::Matrix* U_H_,
    #                      const CAROM::Matrix* Hsinv_, const int myid, const bool oversampling_,
    #                      const bool hyperreduce_, const bool x_base_only_)
    def __init__(self, fom_, fomSp_, rvdim_, rxdim_, hdim_, smm_, v0_,
                 x0_, v0_fom_, V_v_, V_x_, U_H_, Hsinv_, myid, oversampling_,
                 hyperreduce_, x_base_only_):
        mfem.PyTimeDependentOperator.__init__(self, rxdim_ + rvdim_, 0.0)
        self.fom = fom_
        self.fomSp = fomSp_
        self.rxdim, self.rvdim, self.hdim = rxdim_, rvdim_, hdim_
        self.x0, self.v0 = x0_, v0_
        self.v0_fom = v0_fom_
        self.smm = smm_
        self.nsamp_H = smm_.GetNumVarSamples("H")
        self.V_x, self.V_v = V_x_, V_v_
        self.U_H, self.Hsinv = U_H_, Hsinv_
        self.zN = linalg.Vector(max(self.nsamp_H, 1), False)
        self.zX = linalg.Vector(max(self.nsamp_H, 1), False)
        self.oversampling = oversampling_
        self.M_hat_solver = mfem.CGSolver(fom_.fespace.GetComm())
        self.z = mfem.Vector(self.Height() // 2)
        self.hyperreduce = hyperreduce_
        self.x_base_only = x_base_only_

        if (myid == 0):
            self.V_v_sp = linalg.Matrix(self.fomSp.Height() // 2, self.rvdim, False)
            self.V_x_sp = linalg.Matrix(self.fomSp.Height() // 2, self.rxdim, False)
            # TODO(kevin): we might need to initialize for non-root processes as well.

        # Gather distributed vectors
        if (self.x_base_only):
            self.smm.GatherDistributedMatrixRows("X", self.V_v, self.rvdim, self.V_v_sp)
        else:
            self.smm.GatherDistributedMatrixRows("V", self.V_v, self.rvdim, self.V_v_sp)

        self.smm.GatherDistributedMatrixRows("X", self.V_x, self.rxdim, self.V_x_sp)

        # Create V_vTU_H, for hyperreduction
        self.V_vTU_H = self.V_v.transposeMult(self.U_H)

        self.S_hat = linalg.Matrix(self.rvdim, self.rvdim, False)
        self.S_hat_v0 = linalg.Vector(self.rvdim, False)
        self.S_hat_v0_temp = mfem.Vector(self.v0_fom.Size())
        self.S_hat_v0_temp_librom = linalg.Vector(self.S_hat_v0_temp.GetDataArray(), True, False)
        self.M_hat = linalg.Matrix(self.rvdim, self.rvdim, False)
        self.M_hat_inv = linalg.Matrix(self.rvdim, self.rvdim, False)

        # Create S_hat
        ComputeCtAB(self.fom.Smat, self.V_v, self.V_v, self.S_hat)

        # Apply S_hat to the initial velocity and store
        self.fom.Smat.Mult(self.v0_fom, self.S_hat_v0_temp)
        self.V_v.transposeMult(self.S_hat_v0_temp_librom, self.S_hat_v0)

        # Create M_hat
        ComputeCtAB(self.fom.Mmat, self.V_v, self.V_v, self.M_hat)

        # Invert M_hat and store
        self.M_hat.inverse(self.M_hat_inv)

        if (myid == 0):
            self.spdim = self.fomSp.Height()  # Reduced height

            self.zH = mfem.Vector(self.spdim // 2)  # Samples of H

            # Allocate auxillary vectors
            self.z.SetSize(self.spdim // 2)
            self.z_v = mfem.Vector(self.spdim // 2)
            self.z_x = mfem.Vector(self.spdim // 2)
            self.z_librom = linalg.Vector(self.z.GetDataArray(), False, False)
            self.z_v_librom = linalg.Vector(self.z_v.GetDataArray(), False, False)
            self.z_x_librom = linalg.Vector(self.z_x.GetDataArray(), False, False)

            # This is for saving the recreated predictions
            self.psp = mfem.Vector(self.spdim)
            self.psp_librom = linalg.Vector(self.psp.GetDataArray(), False, False)

            # Define sub-vectors of psp.
            self.psp_x = mfem.Vector(self.psp.GetData(), self.spdim // 2)
            self.psp_v = mfem.Vector(self.psp, self.spdim // 2, self.spdim // 2)

            self.psp_v_librom = linalg.Vector(self.psp_v.GetDataArray(), False, False)

        if (not self.hyperreduce):
            fdim = self.fom.Height() # Unreduced height

            self.z.SetSize(fdim // 2)
            self.z_v.SetSize(fdim // 2)
            self.z_x.SetSize(fdim // 2)
            self.z_librom = linalg.Vector(self.z.GetDataArray(), False, False)
            self.z_v_librom = linalg.Vector(self.z_v.GetDataArray(), True, False)
            self.z_x_librom = linalg.Vector(self.z_x.GetDataArray(), True, False)

            # This is for saving the recreated predictions
            self.pfom = mfem.Vector(fdim)
            self.pfom_librom = linalg.Vector(self.pfom.GetDataArray(), False, False)

            # Define sub-vectors of pfom.
            self.pfom_x = mfem.Vector(self.pfom.GetData(), fdim // 2)
            self.pfom_v = mfem.Vector(self.pfom, fdim // 2, fdim // 2)
            self.zfom_x = mfem.Vector(fdim / 2)
            self.zfom_x_librom = linalg.Vector(self.zfom_x.GetDataArray(), True, False)

            self.pfom_v_librom = linalg.Vector(self.pfom_v.GetDataArray(), True, False)

    def Mult(self, vx, dvx_dt):
        if (self.hyperreduce):
            self.Mult_Hyperreduced(vx, dvx_dt)
        else:
            self.Mult_FullOrder(vx, dvx_dt)

    def Mult_Hyperreduced(self, vx, dvx_dt):
        # Check that the sizes match
        assert((vx.Size() == self.rvdim + self.rxdim) and (dvx_dt.Size() == self.rvdim + self.rxdim))

        # Create views to the sub-vectors v, x of vx, and dv_dt, dx_dt of dvx_dt
        v = mfem.Vector(vx, 0, self.rvdim)
        v_librom = linalg.Vector(vx.GetDataArray()[:self.rvdim], False, False)
        x_librom = linalg.Vector(vx.GetDataArray()[self.rvdim:], False, False)
        dv_dt = mfem.Vector(dvx_dt, 0, self.rvdim)
        dx_dt = mfem.Vector(dvx_dt, self.rvdim, self.rxdim)
        dv_dt_librom = linalg.Vector(dv_dt.GetDataArray(), False, False)
        dx_dt_librom = linalg.Vector(dx_dt.GetDataArray(), False, False)

        # Lift the x-, and v-vectors
        # I.e. perform v = v0 + V_v v^, where v^ is the input
        self.V_v_sp.mult(v_librom, self.z_v_librom)
        self.V_x_sp.mult(x_librom, self.z_x_librom)

        add_vector(self.z_v, self.v0, self.psp_v) # Store liftings
        add_vector(self.z_x, self.x0, self.psp_x)

        # Hyperreduce H
        # Apply H to x to get zH
        self.fomSp.H.Mult(self.psp_x, self.zH)

        # Sample the values from zH
        self.smm.GetSampledValues("H", self.zH, self.zN)

        # Apply inverse H-basis
        if (self.oversampling):
            self.Hsinv.transposeMult(self.zN, self.zX)
        else:
            self.Hsinv.mult(self.zN, self.zX)

        # Multiply by V_v^T * U_H
        self.V_vTU_H.mult(self.zX, self.z_librom)

        if (self.fomSp.viscosity != 0.0):
            # Apply S^, the reduced S operator, to v
            self.S_hat.multPlus(self.z_librom, v_librom, 1.0)
            self.z_librom += self.S_hat_v0

        self.z.Neg() # z = -z
        self.M_hat_inv.mult(self.z_librom, dv_dt_librom) # to invert reduced mass matrix operator.

        self.V_x_sp.transposeMult(self.psp_v_librom, dx_dt_librom)

    def Mult_FullOrder(self, vx, dvx_dt):
        # Check that the sizes match
        assert((vx.Size() == self.rvdim + self.rxdim) and (dvx_dt.Size() == self.rvdim + self.rxdim))

        # Create views to the sub-vectors v, x of vx, and dv_dt, dx_dt of dvx_dt
        v = mfem.Vector(vx, 0, self.rvdim)
        v_librom = linalg.Vector(vx.GetDataArray()[:self.rvdim], False, False)
        x_librom = linalg.Vector(vx.GetDataArray()[self.rvdim:], False, False)
        dv_dt = mfem.Vector(dvx_dt, 0, self.rvdim)
        dx_dt = mfem.Vector(dvx_dt, self.rvdim, self.rxdim)
        dv_dt_librom = linalg.Vector(dv_dt.GetDataArray(), False, False)
        dx_dt_librom = linalg.Vector(dx_dt.GetDataArray(), False, False)

        # Lift the x-, and v-vectors
        # I.e. perform v = v0 + V_v v^, where v^ is the input
        self.V_x.mult(x_librom, self.z_x_librom)
        self.V_v.mult(v_librom, self.z_v_librom)

        add_vector(self.z_x, self.x0, self.pfom_x) # Store liftings
        add_vector(self.z_v, self.v0, self.pfom_v)

        # Apply H to x to get z
        self.fom.H.Mult(self.pfom_x, self.zfom_x)

        self.V_v.transposeMult(self.zfom_x_librom, self.z_librom)

        if (self.fom.viscosity != 0.0):
            # Apply S^, the reduced S operator, to v
            self.S_hat.multPlus(self.z_librom, v_librom, 1.0)
            self.z_librom += self.S_hat_v0

        self.z.Neg() # z = -z
        self.M_hat_inv.mult(self.z_librom, dv_dt_librom) # to invert reduced mass matrix operator.

        self.V_x.transposeMult(self.pfom_v_librom, dx_dt_librom)

# /** Function representing the elastic energy density for the given hyperelastic
#     model+deformation. Used in HyperelasticOperator::GetElasticEnergyDensity. */
class ElasticEnergyCoefficient(mfem.PyCoefficient):
    def __init__(self, model, x):
        self.x = x
        self.model = model
        self.J = mfem.DenseMatrix()
        mfem.PyCoefficient.__init__(self)

    def Eval(self, T, ip):
        self.model.SetTransformation(T)
        self.x.GetVectorGradient(T, self.J)
        return self.model.EvalW(self.J)/(self.J.Det())

class InitialDeformationIC1(mfem.VectorPyCoefficient):
    def EvalValue(self, x):
        from copy import deepcopy
        y = deepcopy(x)
        return y

class InitialVelocityIC1(mfem.VectorPyCoefficient):
    def EvalValue(self, x):
        global s
        s_eff = s / 80.0

        v = np.zeros(len(x))
        v[-1] = -s_eff * sin(s * x[0])
        return v

class InitialDeformationIC2(mfem.VectorPyCoefficient):
    def EvalValue(self, x):
        global s
        s_eff = s

        from copy import deepcopy
        y = deepcopy(x)
        y[-1] = x[-1] + 0.25 * x[0] * s_eff
        return y

class InitialVelocityIC2(mfem.VectorPyCoefficient):
    def EvalValue(self, x):
        v = np.zeros(len(x))
        return v


# def GetFirstColumns(N, A):
#     S = linalg.Matrix(A.numRows(), min(N, A.numColumns()), A.distributed())
#     for (int i = 0; i < S->numRows(); ++i)
#     {
#         for (int j = 0; j < S->numColumns(); ++j)
#             (*S)(i, j) = (*A)(i, j);
#     }

#     # delete A;  # TODO: find a good solution for this.
#     return S;

# TODO: move this to the library?
def BasisGeneratorFinalSummary(bg, energyFraction, cutoff, cutoffOutputPath):
    rom_dim = bg.getSpatialBasis().numColumns()
    sing_vals = bg.getSingularValues()

    assert(rom_dim <= sing_vals.dim())

    sum = 0.0
    for sv in range(sing_vals.dim()):
        sum += sing_vals[sv]

    energy_fractions = [ 0.9999999, 0.999999, 0.99999, 0.9999, 0.999, 0.99, 0.9 ]
    reached_cutoff = False

    outfile = open(cutoffOutputPath, 'w')

    partialSum = 0.0
    for sv in range(sing_vals.dim()):
        partialSum += sing_vals[sv]
        for i in range(len(energy_fractions)-1, -1, -1):
            if (partialSum / sum > energy_fractions[i]):
                outfile.write("For energy fraction: %.5E, take first %d of %d basis vectors" % (energy_fractions[i], sv+1, sing_vals.dim()))
                energy_fractions.pop(-1)
            else:
                break

        if ((not reached_cutoff) and (partialSum / sum > energyFraction)):
            cutoff = sv + 1
            reached_cutoff = True

    if (not reached_cutoff): cutoff = sing_vals.dim()
    outfile.write("Take first %d of %d basis vectors" % (cutoff, sing_vals.dim()))
    outfile.close()

    return cutoff

def MergeBasis(dimFOM, nparam, max_num_snapshots, name):
    if (not (nparam > 0)):
        raise RuntimeError("Must specify a positive number of parameter sets")

    update_right_SV = False
    isIncremental = False

    options = linalg.Options(dimFOM, nparam * max_num_snapshots, update_right_SV)
    generator = linalg.BasisGenerator(options, isIncremental, "basis" + name)

    for paramID in range(nparam):
        snapshot_filename = "basis%d_%s_snapshot" % (paramID, name)
        generator.loadSamples(snapshot_filename, "snapshot")

    generator.endSamples() # save the merged basis file

    cutoff = 0
    cutoff = BasisGeneratorFinalSummary(generator, 0.9999999, cutoff, "mergedSV_" + name + ".txt")

# TODO: remove this by making online computation serial?
def BroadcastUndistributedRomVector(v):
    N = v.dim()
    assert(N > 0)

    from copy import deepcopy
    d = deepcopy(v.getData())
    
    from mpi4py import MPI
    MPI.COMM_WORLD.Bcast([d, MPI.DOUBLE], root=0)

    for i in range(N):
        v[i] = d[i]

def visualize(out, mesh, deformed_nodes, field,
              field_name='', init_vis=False):
    nodes = deformed_nodes
    owns_nodes = 0

    nodes, owns_nodes = mesh.SwapNodes(nodes, owns_nodes)

    out.send_text("parallel " + str(mesh.GetNRanks()) + " " + str(mesh.GetMyRank()))
    out.send_solution(mesh, field)

    nodes, owns_nodes = mesh.SwapNodes(nodes, owns_nodes)

    if (init_vis):
        out.send_text("window_size 800 800")
        out.send_text("window_title '" + field_name)
        if (mesh.SpaceDimension() == 2):
            out.send_text("view 0 0")   # view from top
            out.send_text("keys jl")    # turn off perspective and light
        out.send_text("keys cm")         # show colorbar and mesh
        # update value-range; keep mesh-extents fixed
        out.send_text("autoscale value")
        out.send_text("pause")
    out.flush()

# Scaling factor for parameterization
s = 1.0

def run():
    # 1. Initialize MPI.
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    myid = comm.Get_rank()
    num_procs = comm.Get_size()

    # 2. Parse command-line options.
    from mfem.common.arg_parser import ArgParser
    parser = ArgParser(description="Projection ROM - MFEM nonlinear elasticity equation example.")
    parser.add_argument("-m", "--mesh",
                        action='store', dest='mesh_file', default="../data/beam-quad.mesh", type=str,
                        help="Mesh file to use.")
    parser.add_argument("-rs", "--refine-serial",
                        action='store', dest='ser_ref_levels', default=2, type=int,
                        help="Number of times to refine the mesh uniformly in serial.")
    parser.add_argument("-rp", "--refine-parallel",
                        action='store', dest='par_ref_levels', default=0, type=int,
                        help="Number of times to refine the mesh uniformly in parallel.")
    parser.add_argument("-o", "--order",
                        action='store', default=2, type=int,
                        help="Order (degree) of the finite elements.")
    parser.add_argument("-s", "--ode-solver",
                        action='store', dest='ode_solver_type', default=14, type=int,
                        help="ODE solver: 1 - Backward Euler, 2 - SDIRK2, 3 - SDIRK3,\n\t"
                                "            11 - Forward Euler, 12 - RK2,\n\t"
                                "            13 - RK3 SSP, 14 - RK4."
                                "            22 - Implicit Midpoint Method,\n\t"
                                "            23 - SDIRK23 (A-stable), 24 - SDIRK34")
    parser.add_argument("-tf", "--t-final",
                        action='store', default=15.0, type=float,
                        help="Final time; start time is 0.")
    parser.add_argument("-dt", "--time-step",
                        action='store', dest='dt', default=0.03, type=float,
                        help="Time step.")
    parser.add_argument("-v", "--viscosity",
                        action='store', dest='visc', default=1e-2, type=float,
                        help="Viscosity coefficient.")
    parser.add_argument("-mu", "--shear-modulus",
                        action='store', dest='mu', default=0.25, type=float,
                        help="Shear modulus in the Neo-Hookean hyperelastic model.")
    parser.add_argument("-K", "--bulk-modulus",
                        action='store', dest='K', default=5.0, type=float,
                        help="Bulk modulus in the Neo-Hookean hyperelastic model.")
    parser.add_argument("-alrtol", "--adaptive-lin-rtol",
                        action='store_true', default=True, dest='adaptive_lin_rtol',
                        help="Enable adaptive linear solver rtol.")
    parser.add_argument("-no-alrtol", "--no-adaptive-lin-rtol",
                        action='store_false', dest='adaptive_lin_rtol',
                        help="Disable adaptive linear solver rtol.")
    parser.add_argument("-vis", "--visualization",
                        action='store_true', default=True, dest='visualization',
                        help="Enable GLVis visualization.")
    parser.add_argument("-no-vis", "--no-visualization",
                        action='store_false', dest='visualization',
                        help="Disable GLVis visualization.")
    parser.add_argument("-visit", "--visit-datafiles",
                        action='store_true', default=False, dest='visit',
                        help="Save data files for VisIt (visit.llnl.gov) visualization.")
    parser.add_argument("-no-visit", "--no-visit-datafiles",
                        action='store_false', dest='visit',
                        help="Save data files for VisIt (visit.llnl.gov) visualization.")
    parser.add_argument("-vs", "--visualization-steps",
                        action='store', dest='vis_steps', default=1, type=int,
                        help="Visualize every n-th timestep.")
    parser.add_argument("-ns", "--nset",
                        action='store', dest='nsets', default=0, type=int,
                        help="Number of parametric snapshot sets")
    parser.add_argument("-offline", "--offline", 
                        action='store_true', dest='offline', default=False,
                        help="Enable the offline phase.")
    parser.add_argument("-no-offline", "--no-offline",
                        action='store_false', dest='offline',
                        help="Disable the offline phase.")
    parser.add_argument("-online", "--online", 
                        action='store_true', dest='online', default=False,
                        help="Enable the online phase.")
    parser.add_argument("-no-online", "--no-online",
                        action='store_false', dest='online',
                        help="Disable the online phase.")
    parser.add_argument("-merge", "--merge", 
                        action='store_true', dest='merge', default=False,
                        help="Enable the merge phase.")
    parser.add_argument("-no-merge", "--no-merge",
                        action='store_false', dest='merge',
                        help="Disable the merge phase.")
    parser.add_argument("-sopt", "--sopt", 
                        action='store_true', dest='use_sopt', default=False,
                        help="Use S-OPT sampling instead of DEIM for the hyperreduction.")
    parser.add_argument("-no-sopt", "--no-sopt",
                        action='store_false', dest='use_sopt',
                        help="(disable) Use S-OPT sampling instead of DEIM for the hyperreduction.")
    parser.add_argument("-nsr", "--nsr",
                        action='store', dest='num_samples_req', default=-1, type=int,
                        help="number of samples we want to select for the sampling algorithm.")
    parser.add_argument("-rxdim", "--rxdim",
                        action='store', default=-1, type=int,
                        help="Basis dimension for displacement solution space.")
    parser.add_argument("-rvdim", "--rvdim",
                        action='store', default=-1, type=int,
                        help="Basis dimension for velocity solution space.")
    parser.add_argument("-hdim", "--hdim",
                        action='store', default=-1, type=int,
                        help="Basis dimension for the nonlinear term.")
    parser.add_argument("-id", "--id",
                        action='store', dest='id_param', default=0, type=int,
                        help="Parametric index")
    parser.add_argument("-hyp", "--hyperreduce",
                        action='store_true', dest='hyperreduce', default=False,
                        help="Enable Hyper reduce nonlinear term")
    parser.add_argument("-no-hyp", "--no-hyperreduce",
                        action='store_false', dest='hyperreduce',
                        help="Disable Hyper reduce nonlinear term")
    parser.add_argument("-xbo", "--xbase-only",
                        action='store_true', dest='x_base_only', default=False,
                        help="Use the displacement (X) basis to approximate velocity.")
    parser.add_argument("-no-xbo", "--not-xbase-only",
                        action='store_false', dest='x_base_only',
                        help="not use the displacement (X) basis to approximate velocity.")
    parser.add_argument("-def-ic", "--deformation-ic",
                        action='store_true', dest='def_ic', default=False,
                        help="Use a deformation initial condition. Default is velocity IC.")
    parser.add_argument("-vel-ic", "--velocity-ic",
                        action='store_false', dest='def_ic',
                        help="Use velocity initial condition. Default is velocity IC.")
    parser.add_argument("-sc", "--scaling",
                        action='store', dest='s', default=1.0, type=float,
                        help="Scaling factor for initial condition.")

    args = parser.parse_args()
    if (myid == 0): parser.print_options(args)

    mesh_file                   = args.mesh_file
    ser_ref_levels              = args.ser_ref_levels
    par_ref_levels              = args.par_ref_levels
    order                       = args.order
    ode_solver_type             = args.ode_solver_type
    vis_steps                   = args.vis_steps
    t_final                     = args.t_final
    dt                          = args.dt
    visc                        = args.visc
    mu                          = args.mu
    K                           = args.K
    adaptive_lin_rtol           = args.adaptive_lin_rtol
    visualization               = args.visualization
    visit                       = args.visit
    def_ic                      = args.def_ic
    global s
    s                           = args.s

    # ROM parameters
    offline                     = args.offline
    merge                       = args.merge
    online                      = args.online
    use_sopt                    = args.use_sopt
    hyperreduce                 = args.hyperreduce
    x_base_only                 = args.x_base_only
    num_samples_req             = args.num_samples_req

    nsets                       = args.nsets
    id_param                    = args.id_param

    # number of basis vectors to use
    rxdim                       = args.rxdim
    rvdim                       = args.rvdim
    hdim                        = args.hdim

    check = (offline and (not merge) and (not online)) or ((not offline) and merge and (not online)) or ((not offline) and (not merge) and online)
    if not check:
        raise RuntimeError("only one of offline, merge, or online must be true!")

    solveTimer, totalTimer = StopWatch(), StopWatch()
    totalTimer.Start()

    # // 3. Read the serial mesh from the given mesh file on all processors. We can
    # //    handle triangular, quadrilateral, tetrahedral and hexahedral meshes
    # //    with the same code.
    mesh = mfem.Mesh(mesh_file, 1, 1)
    dim = mesh.Dimension()

    # // 4. Define the ODE solver used for time integration. Several implicit
    # //    singly diagonal implicit Runge-Kutta (SDIRK) methods, as well as
    # //    explicit Runge-Kutta methods are available.
    # if ode_solver_type == 1:
    #     ode_solver = BackwardEulerSolver()
    # elif ode_solver_type == 2:
    #     ode_solver = mfem.SDIRK23Solver(2)
    # elif ode_solver_type == 3:
    #     ode_solver = mfem.SDIRK33Solver()
    if ode_solver_type == 11:
        ode_solver = ForwardEulerSolver()
    elif ode_solver_type == 12:
        ode_solver = mfem.RK2Solver(0.5)
    elif ode_solver_type == 13:
        ode_solver = mfem.RK3SSPSolver()
    elif ode_solver_type == 14:
        ode_solver = mfem.RK4Solver()
    elif ode_solver_type == 15:
        ode_solver = mfem.GeneralizedAlphaSolver(0.5)
    # elif ode_solver_type == 22:
    #     ode_solver = mfem.ImplicitMidpointSolver()
    # elif ode_solver_type == 23:
    #     ode_solver = mfem.SDIRK23Solver()
    # elif ode_solver_type == 24:
    #     ode_solver = mfem.SDIRK34Solver()
    else:
        if myid == 0:
            print("Unknown ODE solver type: " + str(ode_solver_type))
        sys.exit()

    # // 5. Refine the mesh in serial to increase the resolution. In this example
    # //    we do 'ser_ref_levels' of uniform refinement, where 'ser_ref_levels' is
    # //    a command-line parameter.
    for lev in range(ser_ref_levels):
        mesh.UniformRefinement()

    # // 6. Define a parallel mesh by a partitioning of the serial mesh. Refine
    # //    this mesh further in parallel to increase the resolution. Once the
    # //    parallel mesh is defined, the serial mesh can be deleted.
    pmesh = mfem.ParMesh(MPI.COMM_WORLD, mesh)
    del mesh

    for lev in range(par_ref_levels):
        pmesh.UniformRefinement()

    # // 7. Define the parallel vector finite element spaces representing the mesh
    # //    deformation x_gf, the velocity v_gf, and the initial configuration,
    # //    x_ref. Define also the elastic energy density, w_gf, which is in a
    # //    discontinuous higher-order space. Since x and v are integrated in time
    # //    as a system, we group them together in block vector vx, on the unique
    # //    parallel degrees of freedom, with offsets given by array true_offset.
    fe_coll = mfem.H1_FECollection(order, dim)
    fespace = mfem.ParFiniteElementSpace(pmesh, fe_coll, dim)
    glob_size = fespace.GlobalTrueVSize()
    if (myid == 0):
        print('Number of velocity/deformation unknowns: ' + str(glob_size))

    true_size = fespace.TrueVSize()
    true_offset = mfem.intArray(3)
    true_offset[0] = 0
    true_offset[1] = true_size
    true_offset[2] = 2*true_size

    vx = mfem.BlockVector(true_offset)

    v_gf = mfem.ParGridFunction(fespace)
    x_gf = mfem.ParGridFunction(fespace)
    v_gf.MakeTRef(fespace, vx, true_offset[0])
    x_gf.MakeTRef(fespace, vx, true_offset[1])
    # v_gf.MakeTRef(&fespace, vx,
    #               true_offset[0]); // Associate a new FiniteElementSpace and new true-dof data with the GridFunction.
    # x_gf.MakeTRef(&fespace, vx, true_offset[1]);

    x_ref = mfem.ParGridFunction(fespace)
    pmesh.GetNodes(x_ref)

    w_fec = mfem.L2_FECollection(order + 1, dim)
    w_fespace = mfem.ParFiniteElementSpace(pmesh, w_fec)
    w_gf = mfem.ParGridFunction(w_fespace)

    # Basis params
    update_right_SV = False
    isIncremental = False
    basisFileName = "basis" + str(id_param)
    max_num_snapshots = int(t_final / dt) + 2

    # The merge phase
    if (merge):
        totalTimer.Reset()
        totalTimer.Start()

        # Merge bases
        if (not x_base_only):
            MergeBasis(true_size, nsets, max_num_snapshots, "V")

        MergeBasis(true_size, nsets, max_num_snapshots, "X")
        MergeBasis(true_size, nsets, max_num_snapshots, "H")

        totalTimer.Stop()
        if (myid == 0):
            print("Elapsed time for merging and building ROM basis: %e second\n" % totalTimer.duration)

        return

    # // 8. Set the initial conditions for v_gf, x_gf and vx, and define the
    # //    boundary conditions on a beam-like mesh (see description above).
    if (def_ic):
        velo = InitialVelocityIC2(dim)
    else:
        velo = InitialVelocityIC1(dim)

    v_gf.ProjectCoefficient(velo)
    v_gf.SetTrueVector()

    if (def_ic):
        deform = InitialDeformationIC2(dim)
    else:
        deform = InitialDeformationIC1(dim)

    x_gf.ProjectCoefficient(deform)
    x_gf.SetTrueVector()

    v_gf.SetFromTrueVector()
    x_gf.SetFromTrueVector()

    v_gf.GetTrueDofs(vx.GetBlock(0))
    x_gf.GetTrueDofs(vx.GetBlock(1))

    ess_bdr = mfem.intArray(fespace.GetMesh().bdr_attributes.Max())
    ess_bdr.Assign(0)
    ess_bdr[0] = 1 # boundary attribute 1 (index 0) is fixed

    ess_tdof_list = mfem.intArray()
    fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list)

    # Store initial vx
    vx0 = mfem.BlockVector(vx)
    vx_diff = mfem.BlockVector(vx)

    # Reduced order solution
    # Vector* wMFEM = 0;
    # CAROM::Vector* w = 0;
    # CAROM::Vector* w_v = 0;
    # CAROM::Vector* w_x = 0;

    # Initialize reconstructed solution
    v_rec = mfem.Vector(v_gf.GetTrueVector())
    x_rec = mfem.Vector(x_gf.GetTrueVector())

    v_rec_librom = linalg.Vector(v_rec.GetDataArray(), True, False)
    x_rec_librom = linalg.Vector(x_rec.GetDataArray(), True, False)

    # // 9. Initialize the hyperelastic operator, the GLVis visualization and print
    # //    the initial energies.
    oper = HyperelasticOperator(fespace, ess_tdof_list, visc, mu, K)
    soper = None

    # Fill dvdt and dxdt
    dvxdt = mfem.Vector(true_size * 2)
    dvdt = mfem.Vector(dvxdt, 0, true_size)
    dxdt = mfem.Vector(dvxdt, true_size, true_size)

    if (visualization):
        vishost = "localhost"
        visport = 19916
        vis_v = mfem.socketstream(vishost, visport)
        vis_v.precision(8)
        visualize(vis_v, pmesh, x_gf, v_gf, "Velocity", True)
        # // Make sure all ranks have sent their 'v' solution before initiating
        # // another set of GLVis connections (one from each rank):
        pmesh.GetComm().Barrier()
        vis_w = mfem.socketstream(vishost, visport)
        if (vis_w.good()):
            oper.GetElasticEnergyDensity(x_gf, w_gf)
            vis_w.precision(8)
            visualize(vis_w, pmesh, x_gf, w_gf, "Elastic energy density", True)

    # // Create data collection for solution output: either VisItDataCollection for
    # // ascii data files, or SidreDataCollection for binary data files.
    dc = None
    if (visit):
        if (offline):
            dc = mfem.VisItDataCollection("nlelast-fom", pmesh)
        else:
            dc = mfem.VisItDataCollection("nlelast-rom", pmesh)

        dc.SetPrecision(8)
        # // To save the mesh using MFEM's parallel mesh format:
        # // dc->SetFormat(DataCollection::PARALLEL_FORMAT);
        dc.RegisterField("x", x_gf)
        dc.RegisterField("v", v_gf)
        dc.SetCycle(0)
        dc.SetTime(0.0)
        dc.Save()

    ee0 = oper.ElasticEnergy(x_gf)
    ke0 = oper.KineticEnergy(v_gf)

    if (myid == 0):
        print("initial elastic energy (EE) = %e" % ee0)
        print("initial kinetic energy (KE) = %e" % ke0)
        print("initial   total energy (TE) = %e" % (ee0 + ke0))

    # // 10. Create pROM object.
    if (offline):
        options = linalg.Options(fespace.GetTrueVSize(), max_num_snapshots, update_right_SV)

        if (not x_base_only):
            basis_generator_v = linalg.BasisGenerator(options, isIncremental, basisFileName + "_V")

        basis_generator_x = linalg.BasisGenerator(options, isIncremental, basisFileName + "_X")

        basis_generator_H = linalg.BasisGenerator(options, isIncremental, basisFileName + "_H")

    # RomOperator* romop = 0;

    # const CAROM::Matrix* BV_librom = 0;
    # const CAROM::Matrix* BX_librom = 0;
    # const CAROM::Matrix* H_librom = 0;
    # const CAROM::Matrix* Hsinv = 0;

    nsamp_H = -1

    # CAROM::SampleMeshManager* smm = nullptr;

    # // 11. Initialize ROM operator
    # // I guess most of this should be done on id =0
    if (online):
        # Read bases
        if (x_base_only):
            readerV = linalg.BasisReader("basisX") # The basis for v uses the x basis instead.
            rvdim = rxdim
        else:
            readerV = linalg.BasisReader("basisV")

        BV_librom = readerV.getSpatialBasis()

        if (rvdim == -1): # Change rvdim
            rvdim = BV_librom.numColumns()
        else:
            BV_librom = BV_librom.getFirstNColumns(rvdim)

        assert(BV_librom.numRows() == true_size)

        if (myid == 0):
            print("reduced V dim = %d\n" % rvdim)

        readerX = linalg.BasisReader("basisX")
        BX_librom = readerX.getSpatialBasis()

        if (rxdim == -1): # Change rxdim
            rxdim = BX_librom.numColumns()
        else:
            BX_librom = BX_librom.getFirstNColumns(rxdim)

        assert(BX_librom.numRows() == true_size)

        if (myid == 0):
            print("reduced X dim = %d\n" % rxdim)

        # Hyper reduce H
        readerH = linalg.BasisReader("basisH")
        H_librom = readerH.getSpatialBasis()

        # Compute sample points
        if (hdim == -1):
            hdim = H_librom.numColumns()

        assert(H_librom.numColumns() >= hdim)

        if (H_librom.numColumns() > hdim):
            H_librom = H_librom.getFirstNColumns(hdim)

        if (myid == 0):
            print("reduced H dim = %d\n" % hdim)

        # vector<int> num_sample_dofs_per_proc(num_procs);

        if (num_samples_req != -1):
            nsamp_H = num_samples_req
        else:
            nsamp_H = hdim

        Hsinv = linalg.Matrix(nsamp_H, hdim, False)
        # vector<int> sample_dofs(nsamp_H);
        if (use_sopt):
            if (myid == 0):
                print("Using S_OPT sampling\n")
            sample_dofs, num_sample_dofs_per_proc = hyper.S_OPT(H_librom, hdim, Hsinv,
                                                                myid, num_procs, nsamp_H)
        elif (nsamp_H != hdim):
            if (myid == 0):
                print("Using GNAT sampling\n")
            sample_dofs, num_sample_dofs_per_proc = hyper.GNAT(H_librom, hdim, Hsinv,
                                                               myid, num_procs, nsamp_H)
        else:
            if (myid == 0):
                print("Using DEIM sampling\n")
            sample_dofs, num_sample_dofs_per_proc = hyper.DEIM(H_librom, hdim, Hsinv,
                                                               myid, num_procs)

        # Construct sample mesh
        nspaces = 1
        spfespace = [None] * nspaces
        spfespace[0] = fespace

        # ParFiniteElementSpace* sp_XV_space;
        smm = mfem_support.SampleMeshManager(spfespace)

        # vector<int> sample_dofs_empty;
        num_sample_dofs_per_proc_empty = [0] * num_procs

        smm.RegisterSampledVariable("V", 0, sample_dofs,
                                    num_sample_dofs_per_proc)
        smm.RegisterSampledVariable("X", 0, sample_dofs,
                                    num_sample_dofs_per_proc)
        smm.RegisterSampledVariable("H", 0, sample_dofs,
                                    num_sample_dofs_per_proc)

        smm.ConstructSampleMesh()

        w = linalg.Vector(rxdim + rvdim, False)
        w_v = linalg.Vector(rvdim, False)
        w_x = linalg.Vector(rxdim, False)
        w.fill(0.0)

        # Note that some of this could be done only on the ROM solver process, but it is tricky, since RomOperator assembles Bsp in parallel.
        wMFEM = mfem.Vector(w.getData(), rxdim + rvdim)

        # Initial conditions
        # Vector*  w_v0 = 0;
        # Vector*  w_x0 = 0;

        sp_size = 0

        if (myid == 0):
            # NOTE(kevin): SampleMeshManager::GetSampleFESpace returns a pointer to a ParFiniteElementSpace,
            #              which is binded via SWIG, not pybind11.
            #              We need a SWIG object 'shell' to wrap the c++ pointer we return from pybind11 side.
            #              The following instantiation is simply creating SWIG object,
            #              to which GetSampleFESpace will return the c++ pointer.
            #              Instantiation can be of any type, since we only need the 'shell'.
            sp_XV_space = mfem.ParFiniteElementSpace(pmesh, fe_coll, dim)
            # NOTE(kevin): Unlike c++ libROM, GetSampleFESpace returns a deep copy.
            smm.GetSampleFESpace(0, sp_XV_space)

            sp_size = sp_XV_space.TrueVSize()
            sp_offset = mfem.intArray(3)
            sp_offset[0] = 0
            sp_offset[1] = sp_size
            sp_offset[2] = 2*sp_size

            # Initialize sp_p with initial conditions.
            sp_vx = mfem.BlockVector(sp_offset)
            sp_v_gf, sp_x_gf = mfem.ParGridFunction(), mfem.ParGridFunction()

            # // 12. Set the initial conditions for v_gf, x_gf and vx, and define the
            # //    boundary conditions on a beam-like mesh (see description above).

            sp_v_gf.MakeTRef(sp_XV_space, sp_vx,
                             sp_offset[0]) # Associate a new FiniteElementSpace and new true-dof data with the GridFunction.
            sp_x_gf.MakeTRef(sp_XV_space, sp_vx, sp_offset[1])

            # VectorFunctionCoefficient* velo = 0;
            # VectorFunctionCoefficient* deform = 0;

            if (def_ic):
                velo = InitialVelocityIC2(dim)
            else:
                velo = InitialVelocityIC1(dim)

            sp_v_gf.ProjectCoefficient(velo)
            sp_v_gf.SetTrueVector()

            if (def_ic):
                deform = InitialDeformationIC2(dim)
            else:
                deform = InitialDeformationIC1(dim)

            sp_x_gf.ProjectCoefficient(deform)
            sp_x_gf.SetTrueVector()

            sp_v_gf.SetFromTrueVector()
            sp_x_gf.SetFromTrueVector()

            # Get initial conditions
            w_v0 = mfem.Vector(sp_v_gf.GetTrueVector())
            w_x0 = mfem.Vector(sp_x_gf.GetTrueVector())

        # Convert essential boundary list from FOM mesh to sample mesh
        # Create binary list where 1 means essential boundary element, 0 means nonessential.
        Ess_mat = linalg.Matrix(true_size, 1, True)
        for i in range(true_size):
            Ess_mat[i,0] = 0.
            for j in range(ess_tdof_list.Size()):
                if (ess_tdof_list[j] == i ):
                    Ess_mat[i,0] = 1.

        # Project binary FOM list onto sampling space
        MPI.COMM_WORLD.bcast(sp_size, root=0)
        Ess_mat_sp = linalg.Matrix(sp_size, 1, False)
        smm.GatherDistributedMatrixRows("X", Ess_mat, 1, Ess_mat_sp)

        # Count number of true elements in new matrix
        num_ess_sp = 0

        for i in range(sp_size):
            if (Ess_mat_sp[i,0] == 1):
                num_ess_sp += 1

        # Initialize essential dof list in sampling space
        ess_tdof_list_sp = mfem.intArray(num_ess_sp)

        # Add indices to list
        ctr = 0
        for i in range(sp_size):
            if (Ess_mat_sp[i,0] == 1):
                ess_tdof_list_sp[ctr] = i
                ctr += 1

        if (myid == 0):
            # Define operator in sample space
            soper = HyperelasticOperator(sp_XV_space, ess_tdof_list_sp, visc, mu, K)

        if (hyperreduce):
            romop = RomOperator(oper, soper, rvdim, rxdim, hdim, smm, w_v0, w_x0,
                                vx0.GetBlock(0), BV_librom, BX_librom, H_librom, Hsinv, myid,
                                (num_samples_req != -1), hyperreduce, x_base_only)
        else:
            romop = RomOperator(oper, soper, rvdim, rxdim, hdim, smm,
                                vx0.GetBlock(0), vx0.GetBlock(1), vx0.GetBlock(0),
                                BV_librom, BX_librom, H_librom, Hsinv, myid,
                                (num_samples_req != -1), hyperreduce, x_base_only)

        # Print lifted initial energies
        BroadcastUndistributedRomVector(w)

        for i in range(rvdim):
            w_v[i] = w[i]

        for i in range(rxdim):
            w_x[i] = w[rvdim + i]

        romop.V_v.mult(w_v, v_rec_librom)
        romop.V_x.mult(w_x, x_rec_librom)

        v_rec += vx0.GetBlock(0)
        x_rec += vx0.GetBlock(1)

        v_gf.SetFromTrueDofs(v_rec)
        x_gf.SetFromTrueDofs(x_rec)

        ee = oper.ElasticEnergy(x_gf)
        ke = oper.KineticEnergy(v_gf)

        if (myid == 0):
            print("Lifted initial energies, EE = %.5e, KE = %.5e, ΔTE = %.5e" % (ee, ke, (ee + ke) - (ee0 + ke0)))

        ode_solver.Init(romop)
    else:
        # fom
        ode_solver.Init(oper)

    # 13. Perform time-integration
    #     (looping over the time iterations, ti, with a time-step dt).
    #     (taking samples and storing it into the pROM object)

    t = 0.0
    ts = []
    oper.SetTime(t)

    last_step = False
    ti = 1
    while (not last_step):
        dt_real = min(dt, t_final - t)

        if (online):
            if (myid == 0):
                solveTimer.Start()
                t, dt = ode_solver.Step(wMFEM, t, dt_real)
                solveTimer.Stop()

            MPI.COMM_WORLD.bcast(t, root=0)
        else:
            solveTimer.Start()
            t, dt = ode_solver.Step(vx, t, dt_real)
            solveTimer.Stop()

        last_step = (t >= t_final - 1e-8 * dt)

        if (offline):
            if (basis_generator_x.isNextSample(t) or (not x_base_only) and basis_generator_v.isNextSample(t)):
                dvxdt = oper.dvxdt_sp.GetData()
                vx_diff = mfem.BlockVector(vx)
                vx_diff -= vx0

            # Take samples
            if ((not x_base_only) and basis_generator_v.isNextSample(t)):
                basis_generator_v.takeSample(vx_diff.GetBlock(0).GetDataArray())
                basis_generator_v.computeNextSampleTime(vx_diff.GetBlock(0).GetDataArray(), dvdt.GetDataArray(), t)
                basis_generator_H.takeSample(oper.H_sp.GetDataArray())

            if (basis_generator_x.isNextSample(t)):
                basis_generator_x.takeSample(vx_diff.GetBlock(1).GetDataArray())
                basis_generator_x.computeNextSampleTime(vx_diff.GetBlock(1).GetDataArray(), dxdt.GetDataArray(), t)

                if (x_base_only):
                    basis_generator_H.takeSample(oper.H_sp.GetDataArray())

        if (last_step or ((ti % vis_steps) == 0)):
            if (online):
                BroadcastUndistributedRomVector(w)

                for i in range(rvdim):
                    w_v[i] = w[i]

                for i in range(rxdim):
                    w_x[i] = w[rvdim + i]

                romop.V_v.mult(w_v, v_rec_librom)
                romop.V_x.mult(w_x, x_rec_librom)

                v_rec += vx0.GetBlock(0)
                x_rec += vx0.GetBlock(1)

                v_gf.SetFromTrueDofs(v_rec)
                x_gf.SetFromTrueDofs(x_rec)

            else:
                v_gf.SetFromTrueVector()
                x_gf.SetFromTrueVector()

            ee = oper.ElasticEnergy(x_gf)
            ke = oper.KineticEnergy(v_gf)

            if (myid == 0):
                print("step %d, t = %f, EE = %.5e, KE = %.5e, ΔTE = %.5e" % (ti, t, ee, ke, (ee + ke) - (ee0 + ke0)))

            if (visualization):
                visualize(vis_v, pmesh, x_gf, v_gf)
                if (vis_w):
                    oper.GetElasticEnergyDensity(x_gf, w_gf)
                    visualize(vis_w, pmesh, x_gf, w_gf)

            if (visit):
                nodes = x_gf
                owns_nodes = 0
                pmesh.SwapNodes(nodes, owns_nodes)

                dc.SetCycle(ti)
                dc.SetTime(t)
                dc.Save()

        ti += 1
    # timestep loop

    if (myid == 0):
        print("Elapsed time for time integration loop %.5e" % solveTimer.duration)

    velo_name = "velocity_s%f.%06d" % (s, myid)
    pos_name = "position_s%f.%06d" % (s, myid)

    if (offline):
        # Sample final solution, to prevent extrapolation in ROM between the last sample and the end of the simulation.
        dvxdt = oper.dvxdt_sp.GetData()
        vx_diff = mfem.BlockVector(vx)
        vx_diff -= vx0

        # Take samples
        if (not x_base_only):
            basis_generator_v.takeSample(vx_diff.GetBlock(0).GetDataArray())
            basis_generator_v.writeSnapshot()
            del basis_generator_v

        basis_generator_H.takeSample(oper.H_sp.GetDataArray())
        basis_generator_H.writeSnapshot()
        del basis_generator_H

        basis_generator_x.takeSample(vx_diff.GetBlock(1).GetDataArray())
        basis_generator_x.writeSnapshot()
        del basis_generator_x

        # 14. Save the displaced mesh, the velocity and elastic energy.
        nodes = x_gf
        owns_nodes = 0
        pmesh.SwapNodes(nodes, owns_nodes)

        mesh_name = "deformed.%06d" % myid
        ee_name = "elastic_energy.%06d" % myid

        pmesh.Print(mesh_name, 8)
        pmesh.SwapNodes(nodes, owns_nodes)

        np.savetxt(velo_name, vx.GetBlock(0).GetDataArray(), fmt='%.16e')
        np.savetxt(pos_name, vx.GetBlock(1).GetDataArray(), fmt='%.16e')

        with open(ee_name, 'w') as fid:
            ee_ofs = io.StringIO()
            ee_ofs.precision = 8
            oper.GetElasticEnergyDensity(x_gf, w_gf)
            w_gf.Save(ee_ofs)
            fid.write(ee_ofs.getvalue())

    # 15. Calculate the relative error between the ROM final solution and the true solution.
    if (online):
        # Initialize FOM solution
        v_fom = mfem.Vector(v_rec.Size())
        x_fom = mfem.Vector(x_rec.Size())

        # Open and load file
        v_fom.Load(velo_name, v_rec.Size())

        x_fom.Load(pos_name, x_rec.Size())

        # Get difference vector
        diff_v = mfem.Vector(v_rec.Size())
        diff_x = mfem.Vector(x_rec.Size())

        subtract_vector(v_rec, v_fom, diff_v)
        subtract_vector(x_rec, x_fom, diff_x)

        # Get norms
        tot_diff_norm_v = sqrt(mfem.InnerProduct(MPI.COMM_WORLD, diff_v, diff_v))
        tot_diff_norm_x = sqrt(mfem.InnerProduct(MPI.COMM_WORLD, diff_x, diff_x))

        tot_v_fom_norm = sqrt(mfem.InnerProduct(MPI.COMM_WORLD, v_fom, v_fom))
        tot_x_fom_norm = sqrt(mfem.InnerProduct(MPI.COMM_WORLD, x_fom, x_fom))

        if (myid == 0):
            print("Relative error of ROM position (x) at t_final: %f is %.8e" % (t_final, tot_diff_norm_x / tot_x_fom_norm))
            print("Relative error of ROM velocity (v) at t_final: %f is %.8e" % (t_final, tot_diff_norm_v / tot_v_fom_norm))

    # 16. Free the used memory.
    del ode_solver
    del pmesh

    totalTimer.Stop()
    if (myid == 0):
        print("Elapsed time for entire simulation %.5e" % totalTimer.duration)

    MPI.Finalize()
    return

if __name__ == "__main__":
    run()
