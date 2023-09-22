#/******************************************************************************
# *
# * Copyright (c) 2013-2023, Lawrence Livermore National Security, LLC
# * and other libROM project developers. See the top-level COPYRIGHT
# * file for details.
# *
# * SPDX-License-Identifier: (Apache-2.0 OR MIT)
# *
# *****************************************************************************/
#
#//                       libROM MFEM Example: DG Advection (adapted from ex9p.cpp)
#//
#// Compile with: make dg_advection
#//
#// =================================================================================
#//
#// Sample runs and results for DMD:
#//
#// Command 1:
#//   mpirun -np 8 python dg_advection.py -p 0 -dt 0.01 -tf 2 -visit
#//
#// Output 1:
#//   Relative error of DMD solution (u) at t_final: 2.0 is 0.00057671379
#//
#// Command 2:
#//   mpirun -np 8 python dg_advection.py -p 3 -rp 1 -dt 0.005 -tf 4 -visit
#//
#// Output 2:
#//   Relative error of DMD solution (u) at t_final: 4.0 is 0.00023390776
#//
#// =================================================================================
#//
#// Description:  This example code solves the time-dependent advection equation
#//               du/dt + v.grad(u) = 0, where v is a given fluid velocity, and
#//               u0(x)=u(0,x) is a given initial condition.
#//
#//               The example demonstrates the use of Discontinuous Galerkin (DG)
#//               bilinear forms in MFEM (face integrators), the use of implicit
#//               and explicit ODE time integrators, the definition of periodic
#//               boundary conditions through periodic meshes, as well as the use
#//               of GLVis for persistent visualization of a time-evolving
#//               solution. Saving of time-dependent data files for visualization
#//               with VisIt (visit.llnl.gov) and ParaView (paraview.org), as
#//               well as the optional saving with ADIOS2 (adios2.readthedocs.io)
#//               are also illustrated.
'''
   MFEM example 9
      This is a version of Example 1 with a simple adaptive mesh
      refinement loop. 
      See c++ version in the MFEM library for more detail 
'''
from mfem import path
import mfem.par as mfem
from mfem.par import intArray
from os.path import expanduser, join, dirname, exists
from mpi4py import MPI
import numpy as np
from numpy import sqrt, pi, cos, sin, hypot, arctan2
from scipy.special import erfc
from mfem.common.arg_parser import ArgParser
from pylibROM.python_utils.StopWatch import StopWatch
from pylibROM.algo import DMD

if __name__ == "__main__":
    num_proc = MPI.COMM_WORLD.size
    myid = MPI.COMM_WORLD.rank
    verbose = (myid == 0)
    

    #- Read in user-supplied options
    parser = ArgParser(description='dg_advection')
    parser.add_argument("-m", "--mesh",
                        default='periodic-hexagon.mesh',
                        action='store',type=str,
                        help="Mesh file to use.")
    parser.add_argument("-p", "--problem",
                        default=0,action='store',type=int,
                        help="Problem setup to use. See options in velocity_function().")
    parser.add_argument("-rs", "--refine_serial",
                        default=2,action='store',type=int,
                        help="Number of times to refine the mesh uniformly in serial.")
    parser.add_argument("-rp", "--refine_parallel",
                        default=0,action='store',type=int,
                        help="Number of times to refine the mesh uniformly in parallel.")
    parser.add_argument("-o", "--order",
                        default=3,action='store',type=int,
                        help="Order (degree) of the finite elements.")
    parser.add_argument("-d", "--device",
                        default='cpu',action='store',type=str,
                        help="Device configuration string, see Device::Configure().")
    parser.add_argument("-s", "--ode_solver",
                        default=4,action='store',type=int,
                        help='''ODE solver: 1 - Forward Euler,\n\t
                                            2 - RK2 SSP, 3 - RK3 SSP, 4 - RK4, 6 - RK6,\n\t''')
    parser.add_argument("-tf", "--t_final",
                        default=10.0,action='store',type=float,
                        help="Final time; start time is 0.")
    parser.add_argument("-dt", "--time_step",
                        default=0.01,action='store',type=float,
                        help="Time step.")
    parser.add_argument("-vis", "--visualization",
                        default=True, action='store_true',
                        help="Enable or disable GLVis visualization.")
    parser.add_argument("-visit", "--visit_datafiles",
                        default=False, action='store_true',
                        help="Save data files for VisIt (visit.llnl.gov) visualization.")
    parser.add_argument("-paraview", "--paraview_datafiles",
                        default=False,action='store_true',
                        help="Save data files for ParaView (paraview.org) visualization.")
    parser.add_argument("-adios2", "--adios2_streams",
                        default=False,action='store_true',
                        help="Save data using adios2 streams.");
    parser.add_argument("-binary", "--binary_datafiles",
                        default=False,action='store_true',
                        help="Use binary (Sidre) or ascii format for VisIt data files.");
    parser.add_argument("-vs", "--visualization_steps",
                        default=5,action='store',type=int,
                        help="Visualize every n-th timestep.")
    parser.add_argument("-ef", "--energy_fraction",
                        default=0.9999,action='store',type=float,
                        help="Energy fraction for DMD.")
    parser.add_argument("-rdim", "--rdim",
                        default=-1,action='store',type=int,
                        help="Reduced dimension for DMD.");
    # assign arguments 
    args                = parser.parse_args()
    mesh_file           = args.mesh
    problem             = args.problem
    ser_ref_levels      = args.refine_serial
    par_ref_levels      = args.refine_parallel
    order               = args.order
    device_config       = args.device
    ode_solver_type     = args.ode_solver
    t_final             = args.t_final
    dt                  = args.time_step
    visualization       = args.visualization
    visit               = args.visit_datafiles
    adios2              = args.adios2_streams
    binary              = args.binary_datafiles
    vis_steps           = args.visualization_steps
    ef                  = args.energy_fraction
    rdim                = args.rdim
    
    device = mfem.Device(device_config)
    if myid == 0:
        device.Print()
    
    # 3. Read the serial mesh from the given mesh file on all processors. We can
    #    handle geometrically periodic meshes in this code.
    meshfile = expanduser(join(dirname(__file__), '..', 'data', mesh_file))
    mesh = mfem.Mesh(meshfile, 1, 1)
    dim = mesh.Dimension()
    
    # 4. Define the ODE solver used for time integration. Several explicit
    #    Runge-Kutta methods are available.
    ode_solver = None
    if ode_solver_type == 1:
        ode_solver = mfem.ForwardEulerSolver()
    elif ode_solver_type == 2:
        ode_solver = mfem.RK2Solver(1.0)
    elif ode_solver_type == 3:
        ode_solver = mfem.RK3SSolver()
    elif ode_solver_type == 4:
        ode_solver = mfem.RK4Solver()
    elif ode_solver_type == 6:
        ode_solver = mfem.RK6Solver()
    else:
        print("Unknown ODE solver type: " + str(ode_solver_type))
        exit
    
    # 5. Refine the mesh to increase the resolution. In this example we do
    #    'ref_levels' of uniform refinement, where 'ref_levels' is a
    #    command-line parameter. If the mesh is of NURBS type, we convert it to
    #    a (piecewise-polynomial) high-order mesh.
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
    
    # 7. Define the discontinuous DG finite element space of the given
    #    polynomial order on the refined mesh.
    fec = mfem.DG_FECollection(order, dim, mfem.BasisType.GaussLobatto)
    fes = mfem.ParFiniteElementSpace(pmesh, fec)
    
    global_vSize = fes.GlobalTrueVSize()
    if myid == 0:
        print("Number of unknowns: " + str(global_vSize))
    
    #
    #  Define coefficient using VecotrPyCoefficient and PyCoefficient
    #  A user needs to define EvalValue method
    #
    
    
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
                return sin(pi * X[0]) * sin(pi * X[1])
    
            return 0.0
    
    # Inflow boundary condition (zero for the problems considered in this example)
    
    
    class inflow_coeff(mfem.PyCoefficient):
        def EvalValue(self, x):
            return 0
    
    # 8. Set up and assemble the bilinear and linear forms corresponding to the
    #    DG discretization. The DGTraceIntegrator involves integrals over mesh
    #    interior faces.
    
    
    velocity = velocity_coeff(dim)
    inflow = inflow_coeff()
    u0 = u0_coeff()
    
    m = mfem.ParBilinearForm(fes)
    m.AddDomainIntegrator(mfem.MassIntegrator())
    k = mfem.ParBilinearForm(fes)
    k.AddDomainIntegrator(mfem.ConvectionIntegrator(velocity, -1.0))
    k.AddInteriorFaceIntegrator(
        mfem.TransposeIntegrator(mfem.DGTraceIntegrator(velocity, 1.0, -0.5)))
    k.AddBdrFaceIntegrator(
        mfem.TransposeIntegrator(mfem.DGTraceIntegrator(velocity, 1.0, -0.5)))
    
    b = mfem.ParLinearForm(fes)
    b.AddBdrFaceIntegrator(
        mfem.BoundaryFlowIntegrator(inflow, velocity, -1.0, -0.5))
    
    m.Assemble()
    m.Finalize()
    skip_zeros = 0
    k.Assemble(skip_zeros)
    k.Finalize(skip_zeros)
    b.Assemble()
    
    M = m.ParallelAssemble()
    K = k.ParallelAssemble()
    B = b.ParallelAssemble()
    
    # 9. Define the initial conditions, save the corresponding grid function to
    #    a file
    u = mfem.ParGridFunction(fes)
    u.ProjectCoefficient(u0)
    U = u.GetTrueDofs()
    
    smyid = '{:0>6d}'.format(myid)
    mesh_name = "dg_advection-mesh."+smyid
    sol_name = "dg_advection-init."+smyid
    pmesh.Print(mesh_name, 8)
    u.Save(sol_name, 8)
    
    
    class FE_Evolution(mfem.PyTimeDependentOperator):
        def __init__(self, M, K, b):
            mfem.PyTimeDependentOperator.__init__(self, M.Height())
    
            self.M_prec = mfem.HypreSmoother()
            self.M_solver = mfem.CGSolver(M.GetComm())
            self.z = mfem.Vector(M.Height())
    
            self.K = K
            self.M = M
            self.b = b
            self.M_prec.SetType(mfem.HypreSmoother.Jacobi)
            self.M_solver.SetPreconditioner(self.M_prec)
            self.M_solver.SetOperator(M)
            self.M_solver.iterative_mode = False
            self.M_solver.SetRelTol(1e-9)
            self.M_solver.SetAbsTol(0.0)
            self.M_solver.SetMaxIter(100)
            self.M_solver.SetPrintLevel(0)
    
    
    #    def EvalMult(self, x):
    #        if you want to impolement Mult in using python objects,
    #        such as numpy.. this needs to be implemented and don't
    #        overwrite Mult
    
    
        def Mult(self, x, y):
            self.K.Mult(x, self.z)
            self.z += b
            self.M_solver.Mult(self.z, y)
    
    # 10. Define the time-dependent evolution operator describing the ODE
    #     right-hand side, and perform time-integration (looping over the time
    #     iterations, ti, with a time-step dt).
    adv = FE_Evolution(M, K, B)

    fom_timer, dmd_training_timer, dmd_prediction_timer = StopWatch(), StopWatch(), StopWatch()

    fom_timer.Start() 
    t = 0.0
    ts = []
    adv.SetTime(t)
    ode_solver.Init(adv)
    fom_timer.Stop()

    # 11. Create DMD object and take initial sample.
    dmd_training_timer.Start()
    dmd_U = DMD(U.Size(),dt)
    dmd_U.takeSample(U.GetDataArray(),t)
    ts.append(t)
    dmd_training_timer.Stop()


    ti = 0
    done = False
    while not done:
        fom_timer.Start()

        dt_real = min(dt,t_final-t)
        t,dt_real = ode_solver.Step(U,t,dt_real)
        ti += 1
        done = (t>= (t_final-1e-8*dt))

        fom_timer.Stop()

        dmd_training_timer.Start()
        dmd_U.takeSample(U.GetDataArray(),t)
        ts.append(t)
        dmd_training_timer.Stop()

        if done or (ti % vis_steps == 0):
            if myid == 0:
                print("time step: " + str(ti) + ", time: " + str(np.round(t, 3)))
            u.Assign(U)
    
    # 12. save the final solution in parallel 
    u.Assign(U)
    sol_name = "dg_advection-final."+smyid
    u.Save(sol_name, 8)

    # 13. Calculate the DMD modes
    if (myid==0) and (rdim != -1) and (ef != -1):
        print('Both rdim and ef are set. ef will be ignored')

    dmd_training_timer.Start()
    if rdim != -1:
        if myid==0:
            print(f'Creating DMD with rdim: {rdim}')
        dmd_U.train(rdim)
    elif ef != -1:
        if myid==0:
            print(f'Creating DMD with energy fraction: {ef}')
        dmd_U.train(ef)
    dmd_training_timer.Stop()
    true_solution_u = mfem.Vector(U.GetDataArray(), U.Size())
    

    # 14. Predict the state at t_final using DMD
    if myid==0:
        print('Predicting solution using DMD')

    dmd_prediction_timer.Start()
    result_u = dmd_U.predict(ts[0])
    initial_dmd_solution_u = mfem.Vector(result_u.getData(),result_u.dim())
    u.SetFromTrueDofs(initial_dmd_solution_u)
    dmd_prediction_timer.Stop()
    result_u = dmd_U.predict(t_final)

    # 15. Calculate the relative error between the DMD final solution and the true solution.
    dmd_solution_u = mfem.Vector(result_u.getData(), result_u.dim())
    diff_u = mfem.Vector(true_solution_u.Size())
    mfem.subtract_vector(dmd_solution_u,true_solution_u,diff_u)

    tot_diff_norm_u = sqrt(mfem.InnerProduct(MPI.COMM_WORLD,diff_u,diff_u))
    tot_true_solution_u_norm = sqrt(mfem.InnerProduct(MPI.COMM_WORLD,true_solution_u,true_solution_u))

    if myid==0:
        error = tot_diff_norm_u/tot_true_solution_u_norm
        print(f'Relative error of DMD solution (u) at t_final: {t_final} is {error:.11f}')
        print(f'Elapsed time for solving FOM: {fom_timer.duration:.6e}')
        print(f'Elapsed time for solving DMD: {dmd_training_timer.duration:.6e}')
        print(f'Elapsed time for predicting DMD: {dmd_prediction_timer.duration:.6e}')
