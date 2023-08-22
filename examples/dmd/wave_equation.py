'''
//
//                       pylibROM PyMFEM Example: Wave_Equation (adapted from ex23.cpp)
//
// Compile with: make wave_equation
//
// =================================================================================
//
// Sample runs and results for Time-Windowing DMD:
//
// Command 1:
//  python3 wave_equation.py -o 4 -tf 5 -nwinsamp 25
//
// Output 1:
// Relative error of DMD solution (u) at t_final: 5.000000 is 1.591E-04
// Elapsed time for solving FOM: 6.374277e+00 second
// Elapsed time for training DMD: 2.605658e+00 second
// Elapsed time for predicting DMD: 2.799344e-02 second
//
// =================================================================================
//
//
// Description:  This example solves the wave equation problem of the form:
//
//                               d^2u/dt^2 = c^2 \Delta u.
//
//               The example demonstrates the use of time dependent operators,
//               implicit solvers and second order time integration.
//
//               We recommend viewing examples on DG Advection and Non-linear Elasticity
//               before viewing this example.
//
'''
import os
import io
import sys
import time
try:
    import mfem.ser as mfem
except ModuleNotFoundError:
    msg = "PyMFEM is not installed yet. Install PyMFEM:\n"
    msg += "\tpip install mfem"
    raise ModuleNotFoundError(msg)

from mfem.ser import intArray
from os.path import expanduser, join, dirname
import numpy as np
from numpy import sin, cos, exp, sqrt, pi, abs, array, floor, log, sum
# pyMFEM does not provide mfem::StopWatch.
# there is not really a similar stopwatch package in python.. (seriously?)
class StopWatch:
    import time
    duration = 0.0
    start_time = 0.0
    stop_time = 0.0
    running = False

    def __init__(self):
        self.Reset()
        return
    
    def Start(self):
        assert(not self.running)
        self.start_time = time.time()
        self.running = True
        return
    
    def Stop(self):
        assert(self.running)
        self.stop_time = time.time()
        self.duration += self.stop_time - self.start_time
        self.running = False
        return
    
    def Reset(self):
        self.duration = 0.0
        self.start_time = 0.0
        self.stop_time = 0.0
        self.running = False
        return

sys.path.append("../../build")
import pylibROM.algo as algo
import pylibROM.linalg as linalg

class WaveOperator(mfem.SecondOrderTimeDependentOperator):
    def __init__(self, fespace, ess_bdr, speed):
        mfem.SecondOrderTimeDependentOperator.__init__(
            self, fespace.GetTrueVSize(), 0.0)

        self.ess_tdof_list = mfem.intArray()

        rel_tol = 1e-8
        fespace.GetEssentialTrueDofs(ess_bdr, self.ess_tdof_list)

        c2 = mfem.ConstantCoefficient(speed*speed)
        K = mfem.BilinearForm(fespace)
        K.AddDomainIntegrator(mfem.DiffusionIntegrator(c2))
        K.Assemble()

        self.Kmat0 = mfem.SparseMatrix()
        self.Kmat = mfem.SparseMatrix()
        dummy = mfem.intArray()
        K.FormSystemMatrix(dummy, self.Kmat0)
        K.FormSystemMatrix(self.ess_tdof_list, self.Kmat)
        self.K = K

        self.Mmat = mfem.SparseMatrix()
        M = mfem.BilinearForm(fespace)
        M.AddDomainIntegrator(mfem.MassIntegrator())
        M.Assemble()
        M.FormSystemMatrix(self.ess_tdof_list, self.Mmat)
        self.M = M

        M_solver = mfem.CGSolver()
        M_prec = mfem.DSmoother()
        M_solver.iterative_mode = False
        M_solver.SetRelTol(rel_tol)
        M_solver.SetAbsTol(0.0)
        M_solver.SetMaxIter(30)
        M_solver.SetPrintLevel(0)
        M_solver.SetPreconditioner(M_prec)
        M_solver.SetOperator(self.Mmat)
        self.M_prec = M_prec
        self.M_solver = M_solver

        T_solver = mfem.CGSolver()
        T_prec = mfem.DSmoother()
        T_solver.iterative_mode = False
        T_solver.SetRelTol(rel_tol)
        T_solver.SetAbsTol(0.0)
        T_solver.SetMaxIter(100)
        T_solver.SetPrintLevel(0)
        T_solver.SetPreconditioner(T_prec)
        self.T_prec = T_prec
        self.T_solver = T_solver
        self.T = None

    def Mult(self, u, du_dt, d2udt2):
        # Compute:
        #    d2udt2 = M^{-1}*-K(u)
        # for d2udt2
        z = mfem.Vector(u.Size())
        self.Kmat.Mult(u, z)
        z.Neg()  # z = -z
        self.M_solver.Mult(z, d2udt2)

    def ImplicitSolve(self, fac0, fac1, u, dudt, d2udt2):
        # Solve the equation:
        #    d2udt2 = M^{-1}*[-K(u + fac0*d2udt2)]
        # for d2udt2
        if self.T is None:
            self.T = mfem.Add(1.0, self.Mmat, fac0, self.Kmat)
            self.T_solver.SetOperator(self.T)
        z = mfem.Vector(u.Size())
        self.Kmat0.Mult(u, z)
        z.Neg()

        # iterate over Array<int> :D
        for j in self.ess_tdof_list:
            z[j] = 0.0

        self.T_solver.Mult(z, d2udt2)

    def SetParameters(self, u):
        self.T = None

class InitialSolution(mfem.PyCoefficient):
    def EvalValue(self, x):
        norm2 = sum(x**2)
        return exp(-norm2*30)

class InitialRate(mfem.PyCoefficient):
    def EvalValue(self, x):
        return 0

if __name__ == "__main__":
    from mfem.common.arg_parser import ArgParser
    parser = ArgParser(description="DMD - MFEM wave equation (ex23) example.")
    parser.add_argument('-m', '--mesh',
                        default='../data/star.mesh',
                        action='store', type=str,
                        help='Mesh file to use.')
    parser.add_argument('-r', '--refine',
                        action='store', default=2, type=int,
                        help="Number of times to refine the mesh uniformly before parallel")
    parser.add_argument('-o', '--order',
                        action='store', default=2, type=int,
                        help="Finite element order (polynomial degree)")
    help_ode = '\n'.join(["ODE solver: [0--10] \t- GeneralizedAlpha(0.1 * s),",
                          "11 \t - Average Acceleration,",
                          "12 \t - Linear Acceleration",
                          "13 \t- CentralDifference",
                          "14 \t- FoxGoodwin"])
    parser.add_argument('-s', '--ode-solver',
                        action='store', default=10, type=int,
                        help=help_ode)
    parser.add_argument('-tf', '--t-final',
                        action='store', default=0.5, type=float,
                        help="Final time; start time is 0.")
    parser.add_argument('-dt', '--time-step',
                        action='store', default=1e-2, type=float,
                        help="Time step")
    parser.add_argument("-c", "--speed",
                        action='store', default=1.0, type=float,
                        help="Wave speed.")
    parser.add_argument("-neu",  "--neumann",
                        action='store_true', default=False,
                        help="BC switch.")
    parser.add_argument('-vis', '--visualization',
                        action='store_true', default=False,
                        help='Enable GLVis visualization')
    parser.add_argument('-visit', '--visit-datafiles',
                        action='store_true', default=False,
                        help="Save data files for VisIt (visit.llnl.gov) visualization.")
    parser.add_argument("-vs", "--visualization-steps",
                        action='store', default=5,  type=int,
                        help="Visualize every n-th timestep.")
    
    parser.add_argument('-rd', '--reference',
                        default='', action='store', type=str,
                        help='Reference directory for checking final solution.')
    parser.add_argument('-ef', '--energy_fraction',
                        action='store', default=0.9999, type=float,
                        help="Energy fraction for DMD.")
    parser.add_argument('-rdim', '--rdim',
                        action='store', default=-1, type=int,
                        help="Reduced dimension for DMD.")
    parser.add_argument('-nwinsamp', '--numwindowsamples',
                        action='store', default=sys.maxsize, type=int,
                        help="Number of samples in DMD windows.")

    args = parser.parse_args()
    parser.print_options(args)
    mesh_file       = expanduser(join(os.path.dirname(__file__),
                                      '..', 'data', args.mesh))
    ref_levels       = args.refine
    order            = args.order
    ode_solver_type  = args.ode_solver
    t_final          = args.t_final
    dt               = args.time_step
    speed            = args.speed
    dirichlet        = (not args.neumann)
    visit            = args.visit_datafiles
    visualization    = args.visualization
    vis_steps        = args.visualization_steps
    ref_dir          = args.reference
    ef               = args.energy_fraction
    rdim             = args.rdim
    windowNumSamples = args.numwindowsamples

    if ((rdim <= 0) and (rdim != -1)):
        raise ValueError("rdim is set to %d, rdim can only be a positive integer or -1" % rdim)

    if (ef <= 0.0):
        raise ValueError("ef must be a positive, it is %f" % ef)
    elif (rdim != -1):
        print("rdim is set to %d" % rdim)

    mesh = mfem.Mesh(mesh_file, 1, 1)
    dim = mesh.Dimension()

    # 3. Define the ODE solver used for time integration. Several second order
    #    time integrators are available.
    if ode_solver_type <= 10:
        ode_solver = mfem.GeneralizedAlpha2Solver(ode_solver_type / 10.)
    elif ode_solver_type == 11:
        ode_solver = mfem.AverageAccelerationSolver()
    elif ode_solver_type == 12:
        ode_solver = mfem.LinearAccelerationSolver()
    elif ode_solver_type == 13:
        ode_solver = mfem.CentralDifferenceSolver()
    elif ode_solver_type == 14:
        ode_solver = mfem.FoxGoodwinSolver()
    else:
        print("Unknown ODE solver type: " + str(ode_solver_type))

    # 4. Refine the mesh to increase the resolution. In this example we do
    #    'ref_levels' of uniform refinement, where 'ref_levels' is a
    #    command-line parameter.
    for lev in range(ref_levels):
        mesh.UniformRefinement()

    # 5. Define the vector finite element space representing the current and the
    #    initial temperature, u_ref.
    fe_coll = mfem.H1_FECollection(order, dim)
    fespace = mfem.FiniteElementSpace(mesh, fe_coll)

    fe_size = fespace.GetTrueVSize()
    print("Number of temperature unknowns: " + str(fe_size))

    u_gf = mfem.GridFunction(fespace)
    dudt_gf = mfem.GridFunction(fespace)

    # 6. Set the initial conditions for u. All boundaries are considered
    #    natural.
    u_0 = InitialSolution()
    dudt_0 = InitialRate()

    u_gf.ProjectCoefficient(u_0)
    u = mfem.Vector()
    u_gf.GetTrueDofs(u)

    dudt_gf.ProjectCoefficient(dudt_0)
    dudt = mfem.Vector()
    dudt_gf.GetTrueDofs(dudt)

    # 7. Initialize the conduction operator and the visualization.
    ess_bdr = mfem.intArray()
    if mesh.bdr_attributes.Size():
        ess_bdr.SetSize(mesh.bdr_attributes.Max())
        if (dirichlet):
            ess_bdr.Assign(1)
        else:
            ess_bdr.Assigne(0)

    oper = WaveOperator(fespace, ess_bdr, speed)

    u_gf.SetFromTrueDofs(u)

    mesh.Print("wave_equation.mesh", 8)
    output = io.StringIO()
    output.precision = 8
    u_gf.Save(output)
    dudt_gf.Save(output)
    fid = open("wave_equation-init.gf", 'w')
    fid.write(output.getvalue())
    fid.close()

    if visit:
        visit_dc = mfem.VisItDataCollection("Wave_Equation", mesh)
        visit_dc.RegisterField("solution", u_gf)
        visit_dc.RegisterField("rate", dudt_gf)
        visit_dc.SetCycle(0)
        visit_dc.SetTime(0.0)
        visit_dc.Save()

    if visualization:
        sout = mfem.socketstream("localhost", 19916)
        if not sout.good():
            print("Unable to connect to GLVis server at localhost:19916")
            visualization = False
            print("GLVis visualization disabled.")
        else:
            sout.precision(precision)
            sout << "solution\n" << mesh << dudt_gf
            sout << "pause\n"
            sout.flush()
            print(
                "GLVis visualization paused. Press space (in the GLVis window) to resume it.")
            
    # 8. Perform time-integration (looping over the time iterations, ti, with a
    #    time-step dt).
    # mfem::StopWatch is not binded by pyMFEM.
    fom_timer, dmd_training_timer, dmd_prediction_timer = StopWatch(), StopWatch(), StopWatch()
    fom_timer.Start()
    ode_solver.Init(oper)
    t = 0.0
    fom_timer.Stop()
    dmd_training_timer.Start()
    curr_window = 0
    ts, dmd_u = [], []
    dmd_u += [algo.DMD(u.Size(), dt)]

    # NOTE: mfem Vector::GetData returns a SWIG Object of type double *.
    # To make it compatible with pybind11, we use ctypes to read data from the memory address.
    from ctypes import *
    uData = (c_double * u.Size()).from_address(int(u.GetData())) # this does not copy the data.
    # uData = list(uData) # this copies the data.
    uData = np.array(uData, copy=False)

    # Showing the memory address info
    print("All of these memory addresses are different.")
    print("id(uData[0]): %d" % id(uData[0]))
    print("int(u.GetData()): %d" % (int(u.GetData()))) # this is not the same as u[0], yet still points to the data.
    print("id(uData): %d" % id(uData))              # this is not the same as u[0], yet still points to the data.

    print("But uData[*] points to the right memory.")
    print("id(u[0]): %d =? id(uData[0]): %d" % (id(u[0]), id(uData[0])))
    print("id(u[1]): %d =? id(uData[1]): %d" % (id(u[1]), id(uData[1])))
    print("uData type: %s" % type(uData))

    dmd_u[curr_window].takeSample(uData, t)
    ts += [t]
    dmd_training_timer.Stop()

    last_step = False
    ti = 0
    while not last_step:
        ti += 1
        if t + dt >= t_final - dt/2:
            last_step = True

        fom_timer.Start()
        t, dt = ode_solver.Step(u, dudt, t, dt)
        fom_timer.Stop()

        dmd_training_timer.Start()
        dmd_u[curr_window].takeSample(uData, t)
        
        if (last_step or (ti % windowNumSamples == 0)):
            print("step %d, t= %f" % (ti, t))

            if (rdim != -1):
                print("Creating DMD with rdim %d at window index: %d" % (rdim, curr_window))
                dmd_u[curr_window].train(rdim)
            else:
                print("Creating DMD with energy fraction: %f at window index: %d" % (ef, curr_window))
                dmd_u[curr_window].train(ef)

            if (not last_step):
                curr_window += 1
                dmd_u += [algo.DMD(u.Size(), dt)]
                dmd_u[curr_window].takeSample(uData, t)
        ts += [t]
        dmd_training_timer.Stop()

        if last_step or (ti % vis_steps == 0):
            print("step " + str(ti) + ", t = " + "{:g}".format(t))

            u_gf.SetFromTrueDofs(u)
            dudt_gf.SetFromTrueDofs(dudt)
            if visualization:
                sout << "solution\n" << mesh << u_gf
                sout.flush()

            if visit:
                visit_dc.SetCycle(ti)
                visit_dc.SetTime(t)
                visit_dc.Save()

        oper.SetParameters(u)

    # 9. Save the final solution. This output can be viewed later using GLVis:
    #    "glvis -m wave_equation.mesh -g wave_equation-final.gf".
    output = io.StringIO()
    output.precision = 8
    u_gf.Save(output)
    dudt_gf.Save(output)
    fid = open("wave_equation-final.gf", 'w')
    fid.write(output.getvalue())
    fid.close()

    # 10. Predict the state at t_final using DMD.
    print("Predicting temperature using DMD")
    dmd_visit_dc = mfem.VisItDataCollection("DMD_Wave_Equation", mesh)
    dmd_visit_dc.RegisterField("solution", u_gf)
    curr_window = 0
    if (visit):
        dmd_prediction_timer.Start()
        result_u = dmd_u[curr_window].predict(ts[0])
        dmd_prediction_timer.Stop()

        # result_u.getData() returns a numpy array, which shares the memory buffer.
        # result_u.getData() does not own the memory.
        initial_dmd_solution_u = mfem.Vector(result_u.getData(), result_u.dim())
        u_gf.SetFromTrueDofs(initial_dmd_solution_u)
        dmd_visit_dc.SetCycle(0)
        dmd_visit_dc.SetTime(0.0)
        dmd_visit_dc.Save()

    for i in range(1, len(ts)):
        if ((i == len(ts) - 1) or (i % vis_steps == 0)):
            if (visit):
                dmd_prediction_timer.Start()
                result_u = dmd_u[curr_window].predict(ts[i])
                dmd_prediction_timer.Stop()

                dmd_solution_u = mfem.Vector(result_u.getData(), result_u.dim())
                u_gf.SetFromTrueDofs(dmd_solution_u)
                dmd_visit_dc.SetCycle(i)
                dmd_visit_dc.SetTime(ts[i])
                dmd_visit_dc.Save()

            if ((i % windowNumSamples == 0) and (i < len(ts)-1)):
                curr_window += 1

    dmd_prediction_timer.Start()
    result_u = dmd_u[curr_window].predict(t_final)
    dmd_prediction_timer.Stop()

    # 11. Calculate the relative error between the DMD final solution and the true solution.
    dmd_solution_u = mfem.Vector(result_u.getData(), result_u.dim())
    
    diff_u = mfem.Vector(u.Size())
    mfem.subtract_vector(dmd_solution_u, u, diff_u)
    tot_diff_norm_u = np.sqrt(mfem.InnerProduct(diff_u, diff_u))
    tot_true_solution_u_norm = np.sqrt(mfem.InnerProduct(u, u))

    print("Relative error of DMD solution (u) at t_final: %f is %.3E" % (t_final, tot_diff_norm_u / tot_true_solution_u_norm))
    print("Elapsed time for solving FOM: %e second\n" % fom_timer.duration)
    print("Elapsed time for training DMD: %e second\n" % dmd_training_timer.duration)
    print("Elapsed time for predicting DMD: %e second\n" % dmd_prediction_timer.duration)