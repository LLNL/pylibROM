'''
/******************************************************************************
 *
 * Copyright (c) 2013-2023, Lawrence Livermore National Security, LLC
 * and other libROM project developers. See the top-level COPYRIGHT
 * file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 *
 *****************************************************************************/

//                       libROM MFEM Example: DG Advection (adapted from ex9p.cpp)
//
// Description:  This example code solves the time-dependent advection equation
//               du/dt + v.grad(u) = 0, where v is a given fluid velocity, and
//               u0(x)=u(0,x) is a given initial condition.
//
//               The example demonstrates the use of Discontinuous Galerkin (DG)
//               bilinear forms in MFEM (face integrators), the use of implicit
//               and explicit ODE time integrators, the definition of periodic
//               boundary conditions through periodic meshes, as well as the use
//               of GLVis for persistent visualization of a time-evolving
//               solution. Saving of time-dependent data files for visualization
//               with VisIt (visit.llnl.gov) and ParaView (paraview.org), as
//               well as the optional saving with ADIOS2 (adios2.readthedocs.io)
//               are also illustrated.
//
// For ROM (reproductive case):
//         python3 dg_advection_local_rom_matrix_interp.py -offline
//         python3 dg_advection_local_rom_matrix_interp.py -online
//
// For ROM (parametric case using matrix interpolation):
//         rm frequencies.txt
//         python3 dg_advection_local_rom_matrix_interp.py -offline -ff 1.02
//         python3 dg_advection_local_rom_matrix_interp.py -interp_prep -ff 1.02 -rdim 40
//         python3 dg_advection_local_rom_matrix_interp.py -offline -ff 1.03
//         python3 dg_advection_local_rom_matrix_interp.py -interp_prep -ff 1.03 -rdim 40
//         python3 dg_advection_local_rom_matrix_interp.py -offline -ff 1.04
//         python3 dg_advection_local_rom_matrix_interp.py -interp_prep -ff 1.04 -rdim 40
//         python3 dg_advection_local_rom_matrix_interp.py -offline -ff 1.06
//         python3 dg_advection_local_rom_matrix_interp.py -interp_prep -ff 1.06 -rdim 40
//         python3 dg_advection_local_rom_matrix_interp.py -offline -ff 1.07
//         python3 dg_advection_local_rom_matrix_interp.py -interp_prep -ff 1.07 -rdim 40
//         python3 dg_advection_local_rom_matrix_interp.py -offline -ff 1.08
//         python3 dg_advection_local_rom_matrix_interp.py -interp_prep -ff 1.08 -rdim 40
//         python3 dg_advection_local_rom_matrix_interp.py -fom -ff 1.05
//         
//     interpolate using linear solve:
//         python3 dg_advection_local_rom_matrix_interp.py -online_interp -ff 1.05 -rdim 40
//     interpolate using lagragian polynomials:
//         python3 dg_advection_local_rom_matrix_interp.py -online_interp -ff 1.05 -rdim 40 -im "LP" 
//     interpolate using inverse distance weighting:
//         python3 dg_advection_local_rom_matrix_interp.py -online_interp -ff 1.05 -rdim 40 -im "IDW"
//
// Sample runs:
//         mpirun -np 4 python3 dg_advection_local_rom_matrix_interp.py -p 0 -dt 0.005
//         mpirun -np 4 python3 dg_advection_local_rom_matrix_interp.py -p 0 -dt 0.01
//         mpirun -np 4 python3 dg_advection_local_rom_matrix_interp.py -p 1 -dt 0.005 -tf 9
//         mpirun -np 4 python3 dg_advection_local_rom_matrix_interp.py -p 1 -rp 1 -dt 0.002 -tf 9
//         mpirun -np 4 python3 dg_advection_local_rom_matrix_interp.py -p 1 -rp 1 -dt 0.02 -s 13 -tf 9
//         mpirun -np 4 python3 dg_advection_local_rom_matrix_interp.py -p 1 -rp 1 -dt 0.004 -tf 9
//         mpirun -np 4 python3 dg_advection_local_rom_matrix_interp.py -p 1 -rp 1 -dt 0.005 -tf 9
//         mpirun -np 4 python3 dg_advection_local_rom_matrix_interp.py -p 3 -rp 2 -dt 0.0025 -tf 9 -vs 20
//         mpirun -np 4 python3 dg_advection_local_rom_matrix_interp.py -p 0 -o 2 -rp 1 -dt 0.01 -tf 8
//         mpirun -np 4 python3 dg_advection_local_rom_matrix_interp.py -p 0 -rs 2 -dt 0.005 -tf 2
//         mpirun -np 4 python3 dg_advection_local_rom_matrix_interp.py -p 0 -rs 1 -o 2 -tf 2
//         mpirun -np 3 python3 dg_advection_local_rom_matrix_interp.py -p 1 -rs 1 -rp 0 -dt 0.005 -tf 0.5
'''

try:
    import mfem.par as mfem
except ModuleNotFoundError:
    msg = "PyMFEM is not installed yet. Install PyMFEM:\n"
    msg += "\tgit clone https://github.com/mfem/PyMFEM.git\n"
    msg += "\tcd PyMFEM\n"
    msg += "\tpython3 setup.py install --with-parallel\n"
    raise ModuleNotFoundError(msg)

import os
import numpy as np
from numpy import arctan2, sqrt, pi, cos, sin, hypot, exp
from scipy.special import erfc
from os.path import expanduser, join, dirname
from ctypes import c_double

import pylibROM.linalg as libROM
from pylibROM.algo.manifold_interp import obtainRotationMatrices, VectorInterpolator, MatrixInterpolator
from pylibROM.python_utils.StopWatch import StopWatch
from pylibROM.mfem import ComputeCtAB, ComputeCtAB_vec


def run():
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    myid = comm.Get_rank()
    num_procs = comm.Get_size()

    from mfem.common.arg_parser import ArgParser
    parser = ArgParser(description="Projection ROM - DG Local Advection.")
    parser.add_argument('-m', '--mesh', dest='mesh_file',
                        default='periodic-hexagon.mesh',
                        action='store', type=str,
                        help='Mesh file to use.')
    parser.add_argument('-p', '--problem', dest='problem',
                        default=3, action='store', type=int,
                        help='Problem setup to use. See options in velocity_function().')
    parser.add_argument('-rs', '--refine-serial', dest='ser_ref_levels',
                        action='store', default=2, type=int,
                        help='Number of times to refine the mesh uniformly in serial.')
    parser.add_argument('-rp', '--refine-parallel', dest='par_ref_levels',
                        action='store', default=0, type=int,
                        help='Number of times to refine the mesh uniformly in parallel.')
    parser.add_argument('-o', '--order', dest='order',
                        action='store', default=3, type=int,
                        help="Finite element order (polynomial degree).")
    parser.add_argument("-pa", "--partial-assembly", dest='pa',
                        action='store_true', default=False,
                        help="Enable Partial Assembly.")
    parser.add_argument("-ea", "--element-assembly", dest='ea',
                        action='store_true', default=False,
                        help="Enable Element Assembly.")
    parser.add_argument("-fa", "--full-assembly", dest='fa',
                        action='store_true', default=False,
                        help="Enable Full Assembly.")
    parser.add_argument("-d", "--device", dest='device_config',
                        action='store', default='cpu', type=str,
                        help="Device configuration string, see Device::Configure().")
    parser.add_argument("-s", "--ode-solver",
                        action='store', dest='ode_solver_type', default=11, type=int,
                        help="ODE solver: 1 - Forward Euler,\n\t"
                                "         2 - RK2 SSP, 3 - RK3 SSP, 4 - RK4, 6 - RK6,\n\t"
                                "         11 - Backward Euler,\n\t"
                                "         12 - SDIRK23 (L-stable), 13 - SDIRK33,\n\t"
                                "         22 - Implicit Midpoint Method,\n\t"
                                "         23 - SDIRK23 (A-stable), 24 - SDIRK34")
    parser.add_argument("-tf", "--t-final", dest='t_final',
                        action='store', default=10.0, type=float,
                        help="Final time; start time is 0.")
    parser.add_argument("-dt", "--time-step",
                        action='store', dest='dt', default=0.01, type=float,
                        help="Time step.")
    parser.add_argument("-ff", "--f-factor",
                        action='store', default=1.0, type=float,
                        help="Frequency scalar factor.")
    parser.add_argument("-cf", "--coefficient",
                        action='store', default=1.0, type=float,
                        help="Coefficient.")
    parser.add_argument("-vis", "--visualization",
                        action='store_true', default=False,
                        help="Enable or disable GLVis visualization.")
    parser.add_argument("-visit", "--visit-datafiles",
                        action='store_true', default=False,
                        help="Save data files for VisIt (visit.llnl.gov) visualization.")
    parser.add_argument("-paraview", "--paraview_datafiles",
                        default=False, action='store_true',
                        help="Save data files for ParaView (paraview.org) visualization.")
    parser.add_argument("-vs", "--visualization-steps", dest='vis_steps',
                        action='store', default=5, type=int,
                        help="Visualize every n-th timestep.")
    parser.add_argument("-fom", "--fom", dest='fom',
                        action='store_true', default=False,
                        help="Enable or disable the fom phase.")
    parser.add_argument("-offline", "--offline", dest='offline',
                        action='store_true', default=False,
                        help="Enable or disable the offline phase.")
    parser.add_argument("-online", "--online", dest='online',
                        action='store_true', default=False,
                        help="Enable or disable the online phase.")
    parser.add_argument("-online_interp", "--online_interp", dest='online_interp',
                        action='store_true', default=False,
                        help="Enable or disable matrix interpolation during the online phase.")
    parser.add_argument("-interp_prep", "--interp_prep", dest='interp_prep',
                        action='store_true', default=False,
                        help="Enable or disable matrix interpolation preparation during the online phase.")
    parser.add_argument("-rt", "--rbf_type", dest='rbf_type',
                        action='store', default='G', type=str,
                        help="RBF type ('G' == gaussian, 'IQ' == inverse quadratic, 'IMQ' == inverse multiquadric).")
    parser.add_argument("-im", "--interp_method", dest='interp_method',
                        action='store', default='LS', type=str,
                        help="Interpolation method ('LS' == linear solve, 'IDW'== inverse distance weighting, 'LP' == lagrangian polynomials).")
    parser.add_argument("-crv", "--crv", dest='closest_rbf_value',
                        action='store', default=0.9, type=float,
                        help="RBF value of the two closest points.")
    parser.add_argument("-ef", "--energy_fraction", dest='ef',
                        action='store', default=0.9999, type=float,
                        help="Energy fraction.")
    parser.add_argument("-rdim", "--rdim", dest='rdim',
                        action='store', default=-1, type=int,
                        help="Reduced dimension.")
    args = parser.parse_args()


    if (myid == 0): parser.print_options(args)

    problem         = args.problem
    ser_ref_levels  = args.ser_ref_levels
    par_ref_levels  = args.par_ref_levels
    order           = args.order
    pa              = args.pa
    ea              = args.ea
    fa              = args.fa
    device_config   = args.device_config
    ode_solver_type = args.ode_solver_type
    t_final         = args.t_final
    dt              = args.dt
    f_factor        = args.f_factor
    visualization   = args.visualization
    visit           = args.visit_datafiles
    paraview        = args.paraview_datafiles
    vis_steps       = args.vis_steps
    fom             = args.fom
    offline         = args.offline
    online          = args.online
    online_interp   = args.online_interp
    interp_prep     = args.interp_prep
    rbf_type        = args.rbf_type
    interp_method   = args.interp_method
    closest_rbf_val = args.closest_rbf_value
    ef              = args.ef
    rdim            = args.rdim

    device = mfem.Device(device_config)
    if myid == 0:
        device.Print()

    check = not (online or interp_prep or online_interp) \
            or (online and not interp_prep and not online_interp) \
            or (not online and ((interp_prep and not online_interp) or (not interp_prep and online_interp)))
    if (not check):
        raise ValueError("only one of online, interp_prep, or online_interp can be true!")

    if interp_prep or online_interp:
        online = True

    # initialize timers
    solveTimer, assembleTimer = StopWatch(), StopWatch()

    # 3. Read the serial mesh from the given mesh file on all processors. We can
    #    handle geometrically periodic meshes in this code.
    mesh_file       = expanduser(join(dirname(__file__), '..', 'data', args.mesh_file))
    if not os.path.exists(mesh_file):
        raise RuntimeError("could not find mesh file %s" % mesh_file)

    mesh = mfem.Mesh(mesh_file, 1, 1)
    dim = mesh.Dimension()

    # 4. Define the ODE solver used for time integration. Several explicit
    #    Runge-Kutta methods are available.
    ode_solver = None
    if ode_solver_type == 1:
        ode_solver = mfem.ForwardEulerSolver()
    elif ode_solver_type == 2:
        ode_solver = mfem.RK2Solver(1.0)
    elif ode_solver_type == 3:
        ode_solver = mfem.RK3SSPSolver()
    elif ode_solver_type == 4:
        ode_solver = mfem.RK4Solver()
    elif ode_solver_type == 6:
        ode_solver = mfem.RK6Solver()
    elif ode_solver_type == 11:
        ode_solver = mfem.BackwardEulerSolver()
    elif ode_solver_type == 12:
        ode_solver = mfem.SDIRK23Solver(2)
    elif ode_solver_type == 13:
        ode_solver = mfem.SDIRK33Solver()
    elif ode_solver_type == 22:
        ode_solver = mfem.ImplicitMidpointSolver()
    elif ode_solver_type == 23:
        ode_solver = mfem.SDIRK23Solver()
    elif ode_solver_type == 24:
        ode_solver = mfem.SDIRK34Solver()
    else:
        if myid == 0:
            print("Unknown ODE solver type: " + str(ode_solver_type))
        exit

    # 5. Refine the mesh in serial to increase the resolution. In this example
    #    we do 'ser_ref_levels' of uniform refinement, where 'ser_ref_levels' is
    #    a command-line parameter. If the mesh is of NURBS type, we convert it
    #    to a (piecewise-polynomial) high-order mesh.
    for lev in range(ser_ref_levels):
        mesh.UniformRefinement()
        if mesh.NURBSext:
            mesh.SetCurvature(max(order, 1))
        bb_min, bb_max = mesh.GetBoundingBox(max(order, 1))

    # 6. Define the parallel mesh by a partitioning of the serial mesh. Refine
    #    this mesh further in parallel to increase the resolution. Once the
    #    parallel mesh is defined, the serial mesh can be deleted.
    pmesh = mfem.ParMesh(MPI.COMM_WORLD, mesh)
    for k in range(par_ref_levels):
        pmesh.UniformRefinement()

    # 7. Define the parallel discontinuous DG finite element space on the
    #    parallel refined mesh of the given polynomial order.
    fec = mfem.DG_FECollection(order, dim, mfem.BasisType.GaussLobatto)
    fes = mfem.ParFiniteElementSpace(pmesh, fec)

    global_vSize = fes.GlobalTrueVSize()
    if myid == 0:
        print("Number of unknowns: " + str(global_vSize))

    class velocity_coeff(mfem.VectorPyCoefficient):
        def EvalValue(self, x):
            dim = len(x)

            center = (bb_min + bb_max)/2.0
            # map to the reference [-1,1] domain
            X = 2 * (x - center) / (bb_max - bb_min)
            if problem == 0:
                if dim == 1:
                    v = [1.0, ]
                elif dim == 2:
                    v = [sqrt(2./3.), sqrt(1./3)]
                elif dim == 3:
                    v = [sqrt(3./6.), sqrt(2./6), sqrt(1./6.)]
            elif (problem == 1 or problem == 2):
                # Clockwise rotation in 2D around the origin
                w = pi/2
                if dim == 1:
                    v = [1.0, ]
                elif dim == 2:
                    v = [w*X[1],  - w*X[0]]
                elif dim == 3:
                    v = [w*X[1],  - w*X[0],  0]
            elif (problem == 3):
                # Clockwise twisting rotation in 2D around the origin
                w = pi/2
                d = max((X[0]+1.)*(1.-X[0]), 0.) * max((X[1]+1.)*(1.-X[1]), 0.)
                d = d ** 2
                if dim == 1:
                    v = [1.0, ]
                elif dim == 2:
                    v = [d*w*X[1],  - d*w*X[0]]
                elif dim == 3:
                    v = [d*w*X[1],  - d*w*X[0],  0]
            return v

    # Initial condition
    class u0_coeff(mfem.PyCoefficient):
        def EvalValue(self, x):
            dim = len(x)

            center = (bb_min + bb_max)/2.0
            # map to the reference [-1,1] domain
            X = 2 * (x - center) / (bb_max - bb_min)
            if (problem == 0 or problem == 1):
                if dim == 1:
                    return exp(-40. * (X[0]-0.5)**2)
                elif (dim == 2 or dim == 3):
                    rx = 0.45
                    ry = 0.25
                    cx = 0.
                    cy = -0.2
                    w = 10.
                    if dim == 3:
                        s = (1. + 0.25*cos(2 * pi * x[2]))
                        rx = rx * s
                        ry = ry * s
                    return (erfc(w * (X[0]-cx-rx)) * erfc(-w*(X[0]-cx+rx)) *
                            erfc(w * (X[1]-cy-ry)) * erfc(-w*(X[1]-cy+ry)))/16

            elif problem == 2:
                rho = hypot(x[0], x[1])
                phi = arctan2(x[1], x[0])
                return (sin(pi * rho) ** 2) * sin(3*phi)
            elif problem == 3:
                return sin(f_factor * pi * X[0]) * sin(f_factor * pi * X[1])

            return 0.0

    # Inflow boundary condition (zero for the problems considered in this example)
    class inflow_coeff(mfem.PyCoefficient):
        def EvalValue(self, x):
            return 0


    # 8. Set up and assemble the parallel bilinear and linear forms (and the
    #    parallel hypre matrices) corresponding to the DG discretization. The
    #    DGTraceIntegrator involves integrals over mesh interior faces.
    
    assembleTimer.Start()
    velocity = velocity_coeff(dim)
    inflow = inflow_coeff()
    u0 = u0_coeff()


    m = mfem.ParBilinearForm(fes)
    k = mfem.ParBilinearForm(fes)

    if pa:
        m.SetAssemblyLevel(mfem.AssemblyLevel_PARTIAL)
        k.SetAssemblyLevel(mfem.AssemblyLevel_PARTIAL)
    elif ea:
        m.SetAssemblyLevel(mfem.AssemblyLevel_ELEMENT)
        k.SetAssemblyLevel(mfem.AssemblyLevel_ELEMENT)
    elif fa:
        m.SetAssemblyLevel(mfem.AssemblyLevel_FULL)
        k.SetAssemblyLevel(mfem.AssemblyLevel_FULL)

    alpha = -1.0
    m.AddDomainIntegrator(mfem.MassIntegrator())
    k.AddDomainIntegrator(mfem.ConvectionIntegrator(velocity, alpha))
    k.AddInteriorFaceIntegrator(
        mfem.NonconservativeDGTraceIntegrator(velocity, alpha))
    k.AddBdrFaceIntegrator(
        mfem.NonconservativeDGTraceIntegrator(velocity, alpha))

    b = mfem.ParLinearForm(fes)
    b.AddBdrFaceIntegrator(
        mfem.BoundaryFlowIntegrator(inflow, velocity, alpha))

    skip_zeros = 0
    m.Assemble()
    k.Assemble(skip_zeros)
    b.Assemble()
    m.Finalize()
    k.Finalize(skip_zeros)
    
    B = b.ParallelAssemble()

    assembleTimer.Stop()

    # 9. Define the initial conditions, save the corresponding grid function to
    #    a file and (optionally) save data in the VisIt format and initialize
    #    GLVis visualization.
    u = mfem.ParGridFunction(fes)
    u.ProjectCoefficient(u0)
    U = u.GetTrueDofs()

    smyid = '{:0>6d}'.format(myid)
    mesh_name = "dg_advection_local_rom_matrix_interp-mesh."+smyid
    sol_name = "dg_advection_local_rom_matrix_interp-init."+smyid
    pmesh.Print(mesh_name, 8)
    u.Save(sol_name, 8)

    # Create data collection for solution output: either VisItDataCollection for
    # ascii data files, or SidreDataCollection for binary data files.
    if visit:
        dc = mfem.VisItDataCollection("DG_Advection", pmesh)
        dc.SetPrecision(8)
        dc.RegisterField("solution", u)
        dc.SetCycle(0)
        dc.SetTime(0.0)
        dc.Save()

    if paraview:
        pd = mfem.ParaViewDataCollection("DG_Advection", pmesh)
        pd.SetPrefixPath("ParaView")
        pd.RegisterField("solution", u)
        pd.SetLevelsOfDetail(order)
        pd.SetDataFormat(mfem.VTKFormat_BINARY)
        pd.SetHighOrderOutput(True)
        pd.SetCycle(0)
        pd.SetTime(0.0)
        pd.Save()

    if visualization:
        MPI.COMM_WORLD.Barrier()
        sout = mfem.socketstream("localhost", 19916)
        if not sout.good():
            visualization = False
            if myid == 0:
                print("Unable to connect to GLVis server at localhost:19916")
                print("GLVis visualization disabled.")
        else:
            sout.send_text("parallel " + str(num_procs) + " " + str(myid))
            sout.precision(8)
            sout.send_solution(pmesh, u)
            sout.send_text("pause")
            sout.flush()
            if myid == 0:
                print("GLVis visualization paused.")
                print(" Press space (in the GLVis window) to resume it.")

    class DG_Solver(mfem.PyIterativeSolver):
        def __init__(self, M, K, fes, comm):
            mfem.PyIterativeSolver.__init__(self, comm)

            self.M = M
            self.K = K
            self.fes = fes
            self.dt = -1.0
            self.A = None

            self.block_size = fes.GetFE(0).GetDof()

            self.prec = mfem.BlockILU(self.block_size, mfem.BlockILU.Reordering_MINIMUM_DISCARDED_FILL)

            self.linear_solver = mfem.GMRESSolver(comm)
            self.linear_solver.iterative_mode = False
            self.linear_solver.SetRelTol(1.0e-9)
            self.linear_solver.SetAbsTol(0.0)
            self.linear_solver.SetMaxIter(100)
            self.linear_solver.SetPrintLevel(0)
            self.linear_solver.SetPreconditioner(self.prec)

            self.M_diag = mfem.SparseMatrix()
            self.M.GetDiag(self.M_diag)

        def SetTimeStep(self, dt):
            if self.dt != dt:
                self.dt = dt

                # Form operator A = M - dt*K
                self.A = mfem.Add(-dt, self.K, 0.0, self.K)

                A_diag = mfem.SparseMatrix()
                self.A.GetDiag(A_diag)
                A_diag.Add(1.0, self.M_diag)

                self.linear_solver.SetOperator(self.A)

        def Mult(self, x, y):
            self.linear_solver.Mult(x, y)

        def SetOperator(self, op):
            self.linear_solver.SetOperator(op)

    class FE_Evolution(mfem.PyTimeDependentOperator):
        def __init__(self, M, K, b):
            mfem.PyTimeDependentOperator.__init__(self, M.Height())

            self.b = b
            self.z = mfem.Vector(M.Height())

            if M.GetAssemblyLevel() == mfem.AssemblyLevel_LEGACY:
                self.M = M.ParallelAssemble()
                self.K = K.ParallelAssemble()
            else:
                self.M = M
                self.K = K

            self.M_solver = mfem.CGSolver(M.ParFESpace().GetComm())
            self.M_solver.SetOperator(self.M)

            if M.GetAssemblyLevel() == mfem.AssemblyLevel_LEGACY:
                self.M_prec = mfem.HypreSmoother(self.M, mfem.HypreSmoother.Jacobi)
                self.dg_solver = DG_Solver(self.M, self.K, M.FESpace(), M.ParFESpace().GetComm())
            else:
                self.M_prec = mfem.OperatorJacobiSmoother()
                self.M_prec.SetOperator(M)
                self.dg_solver = None

            self.M_solver.SetPreconditioner(self.M_prec)
            self.M_solver.iterative_mode = False
            self.M_solver.SetRelTol(1e-9)
            self.M_solver.SetAbsTol(0.0)
            self.M_solver.SetMaxIter(100)
            self.M_solver.SetPrintLevel(0)

        def Mult(self, x, y):
            # M y = K x + b
            self.K.Mult(x, self.z)
            self.z += self.b
            self.M_solver.Mult(self.z, y)

        def ImplicitSolve(self, dt, x, k):
            # (M - dt*K) k = K x + b
            self.K.Mult(x, self.z)
            self.z += self.b
            self.dg_solver.SetTimeStep(dt)
            self.dg_solver.Mult(self.z, k)

    class ROM_FE_Evolution(mfem.PyTimeDependentOperator):
        def __init__(self, M, K, b, u_init_hat, num_cols):
            mfem.PyTimeDependentOperator.__init__(self, num_cols)
            self.z = mfem.Vector(num_cols) 

            self.K = K
            self.M = M
            self.b = b
            self.u_init_hat = u_init_hat

            self.Minv = mfem.DenseMatrixInverse(self.M)
            self.Ainv = None

        def Mult(self, x, y):
            self.K.Mult(x, self.z)
            self.z += self.b
            self.z += self.u_init_hat
            self.Minv.Mult(self.z, y)

        def ImplicitSolve(self, dt, x, k):
            self.K.Mult(x, self.z)
            self.z += self.b
            self.z += self.u_init_hat
        
            # Assume dt is constant. Pre-compute Ainv
            if self.Ainv is None:
                A = mfem.DenseMatrix(self.K.NumRows(), self.K.NumCols())
                A.Set(dt, self.K)
                self.Ainv = mfem.DenseMatrix(self.M)
                self.Ainv -= A
                self.Ainv.Invert()

            self.Ainv.Mult(self.z, k)


    t = 0.0
    max_num_snapshots = int(t_final / dt + 1)
    update_right_SV = False
    isIncremental = False
    basisName = "basis_%f" % (f_factor)
    u_init = mfem.Vector(U)

    # 10. Set BasisGenerator if offline
    if offline:
        options = libROM.Options(fes.GetTrueVSize(), max_num_snapshots,
                            update_right_SV)
        generator = libROM.BasisGenerator(options, isIncremental, basisName)

        u_curr = mfem.Vector(U)
        u_centered = mfem.Vector(U.Size())
        mfem.subtract_vector(u_curr, u_init, u_centered)
        u_centered_vec = np.array((c_double * U.Size()).from_address(int(u_centered.GetData())), copy=False)
        addSample = generator.takeSample(u_centered_vec)
    
    if online:
        if not online_interp:
            reader = libROM.BasisReader(basisName)
            if rdim != -1:
                spatialbasis = reader.getSpatialBasis(rdim)
            else:
                spatialbasis = reader.getSpatialBasis(ef)
            numRowRB = spatialbasis.numRows()
            numColumnRB = spatialbasis.numColumns()
            if (myid == 0):
                print("spatial basis dimension is %d x %d\n" % (numRowRB, numColumnRB))

            M_hat_carom = libROM.Matrix(numColumnRB, numColumnRB, False)
            ComputeCtAB(m, spatialbasis, spatialbasis, M_hat_carom)
            if interp_prep:
                M_hat_carom.write("M_hat_%f" % f_factor)
            M_hat = mfem.DenseMatrix(M_hat_carom.getData())
            # Unlike C++, the conversion from libROM.Matrix to mfem.DenseMatrix does not need tranpose
            # M_hat.Transpose()

            K_hat_carom = libROM.Matrix(numColumnRB, numColumnRB, False)
            ComputeCtAB(k, spatialbasis, spatialbasis, K_hat_carom)
            if interp_prep:
                K_hat_carom.write("K_hat_%f" % f_factor)
            K_hat = mfem.DenseMatrix(K_hat_carom.getData())
            # Unlike C++, the conversion from libROM.Matrix to mfem.DenseMatrix does not need tranpose
            # K_hat.Transpose()

            b_vec = np.array((c_double * B.Size()).from_address(int(B.GetData())), copy=False)
            b_carom = libROM.Vector(b_vec, True)
            b_hat_carom = spatialbasis.transposeMult(b_carom)
            if interp_prep:
                b_hat_carom.write("b_hat_%f" % f_factor)
            b_hat = mfem.Vector(b_hat_carom.getData(), b_hat_carom.dim())

            u_init_hat_carom = libROM.Vector(numColumnRB, False)
            ComputeCtAB_vec(k, U, spatialbasis, u_init_hat_carom)
            if interp_prep:
                u_init_hat_carom.write("U_init_hat_%f" % f_factor)
            u_init_hat = mfem.Vector(u_init_hat_carom.getData(), u_init_hat_carom.dim())

            if interp_prep:
                if myid == 0:
                    with open("frequencies.txt", "a") as f:
                        f.write("%f\n" % f_factor)
                MPI.Finalize()
                return
        else:
            frequencies = np.loadtxt("frequencies.txt")

            parameter_points = []
            bases = []
            M_hats = []
            K_hats = []
            b_hats = []
            u_init_hats = []

            for freq in frequencies:
                point = libROM.Vector(1, False)
                point[0] = freq
                parameter_points.append(point)

                parametricBasisName = "basis_%f" % freq
                reader = libROM.BasisReader(parametricBasisName)

                if rdim == -1:
                    raise RuntimeError("rdim must be used for interpolation.")

                parametricSpatialBasis = reader.getSpatialBasis(rdim)
                numRowRB = parametricSpatialBasis.numRows()
                numColumnRB = parametricSpatialBasis.numColumns()
                bases.append(parametricSpatialBasis)

                parametricMhat = libROM.Matrix()
                parametricMhat.read("M_hat_%f" % freq)
                M_hats.append(parametricMhat)

                parametricKhat = libROM.Matrix()
                parametricKhat.read("K_hat_%f" % freq)
                K_hats.append(parametricKhat)

                parametricbhat = libROM.Vector()
                parametricbhat.read("b_hat_%f" % freq)
                b_hats.append(parametricbhat)

                parametricuinithat = libROM.Vector()
                parametricuinithat.read("u_init_hat_%f" % freq)
                u_init_hats.append(parametricuinithat)
            
            if myid == 0:
                print("spatial basis dimension is %d x %d" % (numRowRB, numColumnRB))

            curr_point = libROM.Vector(1, False)
            curr_point[0] = f_factor

            ref_point = libROM.getClosestPoint(parameter_points, curr_point)
            rotation_matrices = obtainRotationMatrices(parameter_points, bases, ref_point)

            basis_interpolator = MatrixInterpolator(parameter_points,
                    rotation_matrices, bases, ref_point, "B", rbf_type, interp_method,
                    closest_rbf_val)
            M_interpolator = MatrixInterpolator(parameter_points, rotation_matrices,
                    M_hats, ref_point, "SPD", rbf_type, interp_method, closest_rbf_val)
            K_interpolator = MatrixInterpolator(parameter_points, rotation_matrices,
                    K_hats, ref_point, "R", rbf_type, interp_method, closest_rbf_val)
            b_interpolator = VectorInterpolator(parameter_points, rotation_matrices,
                    b_hats, ref_point, rbf_type, interp_method, closest_rbf_val)
            u_init_interpolator = VectorInterpolator(parameter_points,
                    rotation_matrices, u_init_hats, ref_point, rbf_type, interp_method,
                    closest_rbf_val)
            spatialbasis = basis_interpolator.interpolate(curr_point)
            M_hat_carom = M_interpolator.interpolate(curr_point)
            K_hat_carom = K_interpolator.interpolate(curr_point)
            b_hat_carom = b_interpolator.interpolate(curr_point)
            u_init_hat_carom = u_init_interpolator.interpolate(curr_point)

            # Unlike C++, the conversion from libROM.Matrix to mfem.DenseMatrix does not need tranpose
            M_hat = mfem.DenseMatrix(M_hat_carom.getData())
            K_hat = mfem.DenseMatrix(K_hat_carom.getData())

            b_hat = mfem.Vector(b_hat_carom.getData(), b_hat_carom.dim())
            u_init_hat = mfem.Vector(u_init_hat_carom.getData(), u_init_hat_carom.dim())


        u_in = mfem.Vector(numColumnRB)
        u_in.Assign(0.0)


    # 10. Define the time-dependent evolution operator describing the ODE
    #     right-hand side, and perform time-integration (looping over the time
    #     iterations, ti, with a time-step dt).

    if online:
        adv = ROM_FE_Evolution(M_hat, K_hat, b_hat, u_init_hat, numColumnRB)
    else:
        adv = FE_Evolution(m, k, B)
    adv.SetTime(t)
    ode_solver.Init(adv)
    # assembleTimer.Stop()

    # 13. Time marching
    ti = 0
    done = False
    while not done:
        dt_real = min(dt, t_final - t)
        solveTimer.Start()
        if online:
            t, dt = ode_solver.Step(u_in, t, dt_real)
        else:
            t, dt = ode_solver.Step(U, t, dt_real)
        solveTimer.Stop()
        ti = ti + 1

        if t >= t_final - 1e-8 * dt:
            done = True

        # take and write snapshot for ROM
        if offline:
            u_curr = mfem.Vector(U)
            u_centered = mfem.Vector(U.Size())
            mfem.subtract_vector(u_curr, u_init, u_centered)
            u_centered_vec = np.array((c_double * u_centered.Size()).from_address(int(u_centered.GetData())), copy=False)
            addSample = generator.takeSample(u_centered_vec)

        if done or ti % vis_steps == 0:
            if myid == 0:
                print("time step: " + str(ti) + ", time: " + str(np.round(t, 3)))

            # 11. Extract the parallel grid function corresponding to the finite
            #     element approximation U (the local solution on each processor).
            u.Distribute(U)

            if visualization:
                sout.send_text("parallel " + str(num_procs) + " " + str(myid))
                sout.send_solution(pmesh, u)
                sout.flush()

            if visit:
                dc.SetCycle(ti)
                dc.SetTime(t)
                dc.Save()

            if paraview:
                pd.SetCycle(ti)
                pd.SetTime(t)
                pd.Save()

    # 14. Compute basis
    if offline:
        generator.endSamples()
        del generator
        del options

    # 15. Save and compare solution
    solution_filename_fom = "dg_advection_local_rom_matrix_interp-final.%f.%06d" % (f_factor, myid)
    if online:
        u_hat_final_vec = np.array((c_double * u_in.Size()).from_address(int(u_in.GetData())), copy=False)
        u_hat_final_carom = libROM.Vector(u_hat_final_vec, False)
        u_final_carom = libROM.Vector(U.Size(), True)
        spatialbasis.mult(u_hat_final_carom, u_final_carom)
        u_final = mfem.Vector(u_final_carom.getData(), u_final_carom.dim())
        u_final += u_init

        fom_solution = mfem.Vector(U.Size())
        fom_solution.Load(solution_filename_fom, U.Size())
        fomNorm = np.sqrt(mfem.InnerProduct(MPI.COMM_WORLD, fom_solution, fom_solution))
        diff_solution = mfem.Vector(u_final.Size())
        mfem.subtract_vector(fom_solution, u_final, diff_solution)
        diffNorm = np.sqrt(mfem.InnerProduct(MPI.COMM_WORLD, diff_solution, diff_solution))
        if myid == 0:
            print("Relative L2 error of ROM solution = %.5E" % (diffNorm / fomNorm))
            print("Elapsed time for assembling ROM: %e second\n" % assembleTimer.duration)
            print("Elapsed time for solving ROM: %e second\n" % solveTimer.duration)

    # 12. Save the final solution in parallel. This output can be viewed later
    #     using GLVis: "glvis -np <np> -m dg_advection_local_rom_matrix_interp-mesh -g dg_advection_local_rom_matrix_interp-final".
    if offline or fom:
        u.Distribute(U)
        u_sol = np.array((c_double * U.Size()).from_address(int(U.GetData())), copy=False)
        np.savetxt(solution_filename_fom, u_sol, fmt='%.16f')
        if myid == 0:
            print("Elapsed time for assembling FOM: %e second\n" % assembleTimer.duration)
            print("Elapsed time for solving FOM: %e second\n" % solveTimer.duration)

    MPI.Finalize()
    return


if __name__ == "__main__":
    run()
