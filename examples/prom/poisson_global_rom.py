'''
   MFEM example 23
      See c++ version in the MFEM library for more detail 
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

from ctypes import c_double
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
import pylibROM.linalg as libROM

def run():
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    myid = comm.Get_rank()
    num_procs = comm.Get_size()

    from mfem.common.arg_parser import ArgParser
    parser = ArgParser(description="Projection ROM - MFEM Poisson equation example.")
    parser.add_argument('-m', '--mesh',
                        default='../data/star.mesh',
                        action='store', type=str,
                        help='Mesh file to use.')
    parser.add_argument('-o', '--order',
                        action='store', default=1, type=int,
                        help="Finite element order (polynomial degree) or -1 for isoparametric space.")
    parser.add_argument("-id", "--id",
                        action='store', default=0, type=int, help="Parametric id")
    parser.add_argument("-ns", "--nset",
                        action='store', default=0, type=int, help="Number of parametric snapshot sets")
    parser.add_argument("-sc", "--static-condensation",
                        action='store_true', default=False,
                        help="Enable static condensation.")
    parser.add_argument("-pa", "--partial-assembly",
                        action='store_true', default=False,
                        help="Enable Partial Assembly.")
    parser.add_argument("-f", "--frequency",
                        action='store', default=1.0, type=float,
                        help="Set the frequency for the exact solution.")
    parser.add_argument("-cf", "--coefficient",
                        action='store', default=1.0, type=float,
                        help="Coefficient.")
    parser.add_argument("-d", "--device",
                        action='store', default='cpu', type=str,
                        help="Device configuration string, see Device::Configure().")
    parser.add_argument("-visit", "--visit-datafiles",
                        action='store_true', default=True,
                        help="Save data files for VisIt (visit.llnl.gov) visualization.")
    parser.add_argument("-vis", "--visualization",
                        action='store_true', default=True,
                        help="Enable or disable GLVis visualization.")
    parser.add_argument("-fom", "--fom",
                        action='store_true', default=False,
                        help="Enable or disable the fom phase.")
    parser.add_argument("-offline", "--offline",
                        action='store_true', default=False,
                        help="Enable or disable the offline phase.")
    parser.add_argument("-online", "--online",
                        action='store_true', default=False,
                        help="Enable or disable the online phase.")
    parser.add_argument("-merge", "--merge",
                        action='store_true', default=False,
                        help="Enable or disable the merge phase.")

    args = parser.parse_args()
    parser.print_options(args)

    mesh_file       = expanduser(join(os.path.dirname(__file__),
                                      '..', 'data', args.mesh))
    freq            = args.frequency
    fom             = args.fom
    offline         = args.offline
    online          = args.online
    merge           = args.merge
    device_config   = args.device
    id              = args.id
    order           = args.order
    nsets           = args.nset
    coef            = args.coefficient
    pa              = args.partial_assembly
    static_cond     = args.static_condensation

    # ref_levels       = args.refine
    # ode_solver_type  = args.ode_solver
    # t_final          = args.t_final
    # dt               = args.time_step
    # speed            = args.speed
    # dirichlet        = (not args.neumann)
    # visit            = args.visit_datafiles
    # visualization    = args.visualization
    # vis_steps        = args.visualization_steps
    # ref_dir          = args.reference
    # ef               = args.energy_fraction
    # rdim             = args.rdim
    # windowNumSamples = args.numwindowsamples

    kappa = freq * np.pi
    if (fom):
        if (not (fom and (not offline) and (not online))):
            raise ValueError("offline and online must be turned off if fom is used.")
    else:
        check = (offline and (not merge) and (not online))          \
                or ((not offline) and merge and (not online))       \
                or ((not offline) and (not merge) and online)
        if (not check):
            raise ValueError("only one of offline, merge, or online must be true!")
        
    # 3. Enable hardware devices such as GPUs, and programming models such as
    #    CUDA, OCCA, RAJA and OpenMP based on command line options.
    device = mfem.Device(device_config)
    if (myid == 0):
        device.Print()

    # 4. Read the (serial) mesh from the given mesh file on all processors.  We
    #    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
    #    and volume meshes with the same code.
    mesh = mfem.Mesh(mesh_file, 1, 1)
    dim = mesh.Dimension()

    # 5. Refine the serial mesh on all processors to increase the resolution. In
    #    this example we do 'ref_levels' of uniform refinement. We choose
    #    'ref_levels' to be the largest number that gives a final mesh with no
    #    more than 10,000 elements.
    ref_levels = int(np.floor(np.log(10000. / mesh.GetNE()) / log(2.) / dim))
    for l in range(ref_levels):
        mesh.UniformRefinement()

    # 6. Define a parallel mesh by a partitioning of the serial mesh. Refine
    #    this mesh further in parallel to increase the resolution. Once the
    #    parallel mesh is defined, the serial mesh can be deleted.
    # TODO(kevin): figure out mfem parallel version install
    # pmesh = mfem.ParMesh(MPI_COMM_WORLD, mesh)
    pmesh = mfem.Mesh(mesh)
    mesh.Clear()
    par_ref_levels = 2
    for l in range(par_ref_levels):
        pmesh.UniformRefinement()

    # 7. Define a parallel finite element space on the parallel mesh. Here we
    #    use continuous Lagrange finite elements of the specified order. If
    #    order < 1, we instead use an isoparametric/isogeometric space.
    if (order > 0):
        fec = mfem.H1_FECollection(order, dim)
        delete_fec = True
    elif (pmesh.GetNodes()):
        fec = pmesh.GetNodes().OwnFEC()
        delete_fec = False
        if (myid == 0):
            print("Using isoparametric FEs: %s" % fec.Name())
    else:
        fec = mfem.H1_FECollection(1, dim)
        delete_fec = True

    # TODO(kevin): figure out mfem parallel version install
    # fespace = mfem.ParFiniteElementSpace(pmesh, fec)
    # size = fespace.GlobalTrueVSize()
    fespace = mfem.FiniteElementSpace(pmesh, fec)
    size = fespace.GetTrueVSize()
    if (myid == 0):
        print("Number of finite element unknowns: %d" % size)

    # 8. Determine the list of true (i.e. parallel conforming) essential
    #    boundary dofs. In this example, the boundary conditions are defined
    #    by marking all the boundary attributes from the mesh as essential
    #    (Dirichlet) and converting them to a list of true dofs.
    ess_tdof_list = mfem.intArray()
    if (pmesh.bdr_attributes.Size() > 0):
        ess_bdr = mfem.intArray(pmesh.bdr_attributes.Max())
        ess_bdr.Assign(1)
        fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list)

    # 9. Initiate ROM related variables
    max_num_snapshots = 100
    update_right_SV = False
    isIncremental = False
    basisName = "basis"
    basisFileName = "%s%d" % (basisName, id)
    # const CAROM::Matrix* spatialbasis
    # CAROM::Options* options;
    # CAROM::BasisGenerator *generator;
    # int numRowRB, numColumnRB;
    # StopWatch solveTimer, assembleTimer, mergeTimer;

    # 10. Set BasisGenerator if offline
    if (offline):
        options = libROM.Options(fespace.GetTrueVSize(), max_num_snapshots, 1,
                                update_right_SV)
        generator = libROM.BasisGenerator(options, isIncremental, basisFileName)

    # 11. The merge phase
    if (merge):
        # mergeTimer.Start();
        options = libROM.Options(fespace.GetTrueVSize(), max_num_snapshots, 1,
                                update_right_SV)
        generator = libROM.BasisGenerator(options, isIncremental, basisName)
        for paramID in range(nsets):
            snapshot_filename = "%s%d_snapshot" % (basisName, paramID)
            generator.loadSamples(snapshot_filename,"snapshot", 5)
        
        generator.endSamples() # save the merged basis file
        # mergeTimer.Stop();
        # if (myid == 0):
        #     print("Elapsed time for merging and building ROM basis: %e second\n" %
        #            mergeTimer.RealTime())
        del generator
        del options
        MPI.Finalize()
        return
    
    # 12. Set up the parallel linear form b(.) which corresponds to the
    #     right-hand side of the FEM linear system, which in this case is
    #     (f,phi_i) where f is given by the function f_exact and phi_i are the
    #     basis functions in the finite element fespace.
    # assembleTimer.Start();
    # TODO(kevin): figure out mfem parallel install
    # b = mfem.ParLinearForm(fespace)
    class RightHandSide(mfem.PyCoefficient):
        def EvalValue(self, x):
            if (dim == 3):
                return sin(kappa * (x[0] + x[1] + x[2]))
            else:
                return sin(kappa * (x[0] + x[1]))
    b = mfem.LinearForm(fespace)
    f = RightHandSide()
    b.AddDomainIntegrator(mfem.DomainLFIntegrator(f))
    b.Assemble()

    # 13. Define the solution vector x as a parallel finite element grid function
    #     corresponding to fespace. Initialize x with initial guess of zero,
    #     which satisfies the boundary conditions.
    # TODO(kevin): figure out mfem parallel install
    # x = mfem.ParGridFunction(fespace)
    x = mfem.GridFunction(fespace)
    x.Assign(0.0)

    # 14. Set up the parallel bilinear form a(.,.) on the finite element space
    #     corresponding to the Laplacian operator -Delta, by adding the Diffusion
    #     domain integrator.
    # TODO(kevin): figure out mfem parallel install
    # a = mfem.ParBilinearForm(fespace)
    a = mfem.BilinearForm(fespace)
    one = mfem.ConstantCoefficient(coef)
    # if (pa):
    #     a.SetAssemblyLevel(mfem.AssemblyLevel.PARTIAL)
    a.AddDomainIntegrator(mfem.DiffusionIntegrator(one))

    # 15. Assemble the parallel bilinear form and the corresponding linear
    #     system, applying any necessary transformations such as: parallel
    #     assembly, eliminating boundary conditions, applying conforming
    #     constraints for non-conforming AMR, static condensation, etc.
    if (static_cond):
        a.EnableStaticCondensation()
    a.Assemble()

    A = mfem.OperatorPtr()
    B = mfem.Vector()
    X = mfem.Vector()
    a.FormLinearSystem(ess_tdof_list, x, b, A, X, B)
    # assembleTimer.Stop();

    # 16. The offline phase
    if (fom or offline):
        # 17. Solve the full order linear system A X = B
        prec = None
        if pa:
            if mfem.UsesTensorBasis(fespace):
                prec = mfem.OperatorJacobiSmoother(a, ess_tdof_list)
            # TODO(kevin): figure out mfem parallel install
            # else:
                # prec = mfem.HypreBoomerAMG()

        # TODO(kevin): figure out mfem parallel install
        # cg = mfem.CGSolver(MPI.COMM_WORLD)
        cg = mfem.CGSolver()
        cg.SetRelTol(1e-12)
        cg.SetMaxIter(2000)
        cg.SetPrintLevel(1)
        if (prec is not None):
            cg.SetPreconditioner(prec)
        cg.SetOperator(A.Ptr())
        # solveTimer.Start();
        cg.Mult(B, X)
        # solveTimer.Stop();
        if (prec is not None):
            del prec

        # 18. take and write snapshot for ROM
        if (offline):
            # NOTE: mfem Vector::GetData returns a SWIG Object of type double *.
            # To make it compatible with pybind11, we use ctypes to read data from the memory address.
            xData = np.array((c_double * X.Size()).from_address(int(X.GetData())), copy=False) # this does not copy the data.
            addSample = generator.takeSample(xData, 0.0, 0.01)
            generator.writeSnapshot()
            del generator
            del options

    # 19. The online phase
    if (online):
        # 20. read the reduced basis
        # assembleTimer.Start();
        reader = libROM.BasisReader(basisName)
        spatialbasis = reader.getSpatialBasis(0.0)
        numRowRB = spatialbasis.numRows()
        numColumnRB = spatialbasis.numColumns()
        if (myid == 0):
            print("spatial basis dimension is %d x %d\n" % (numRowRB, numColumnRB))

        # libROM stores the matrix row-wise, so wrapping as a DenseMatrix in MFEM means it is transposed.
        reducedBasisT = mfem.DenseMatrix(spatialbasis.getData(),
                                        numColumnRB, numRowRB)

        # 21. form inverse ROM operator
        invReducedA = libROM.Matrix(numColumnRB, numColumnRB, False)
        libROM.ComputeCtAB(A, spatialbasis, spatialbasis, invReducedA)
        invReducedA.inverse()

        bData = np.array((c_double * B.Size()).from_address(int(B.GetData())), copy=False)
        B_carom = libROM.Vector(bData, B.Size(), True, False)
        bData = np.array((c_double * X.Size()).from_address(int(X.GetData())), copy=False)
        X_carom = libROM.Vector(xData, X.Size(), True, False)
        reducedRHS = spatialbasis.transposeMult(B_carom)
        reducedSol = libROM.Vector(numColumnRB, False)
        # assembleTimer.Stop();

        # 22. solve ROM
        # solveTimer.Start();
        invReducedA.mult(reducedRHS, reducedSol)
        # solveTimer.Stop();

        # 23. reconstruct FOM state
        spatialbasis.mult(reducedSol, X_carom)
        del spatialbasis
        del reducedRHS

    # if ((rdim <= 0) and (rdim != -1)):
    #     raise ValueError("rdim is set to %d, rdim can only be a positive integer or -1" % rdim)

    # if (ef <= 0.0):
    #     raise ValueError("ef must be a positive, it is %f" % ef)
    # elif (rdim != -1):
    #     print("rdim is set to %d" % rdim)

    # mesh = mfem.Mesh(mesh_file, 1, 1)
    # dim = mesh.Dimension()

    # # 3. Define the ODE solver used for time integration. Several second order
    # #    time integrators are available.
    # if ode_solver_type <= 10:
    #     ode_solver = mfem.GeneralizedAlpha2Solver(ode_solver_type / 10.)
    # elif ode_solver_type == 11:
    #     ode_solver = mfem.AverageAccelerationSolver()
    # elif ode_solver_type == 12:
    #     ode_solver = mfem.LinearAccelerationSolver()
    # elif ode_solver_type == 13:
    #     ode_solver = mfem.CentralDifferenceSolver()
    # elif ode_solver_type == 14:
    #     ode_solver = mfem.FoxGoodwinSolver()
    # else:
    #     print("Unknown ODE solver type: " + str(ode_solver_type))

    # # 4. Refine the mesh to increase the resolution. In this example we do
    # #    'ref_levels' of uniform refinement, where 'ref_levels' is a
    # #    command-line parameter.
    # for lev in range(ref_levels):
    #     mesh.UniformRefinement()

    # # 5. Define the vector finite element space representing the current and the
    # #    initial temperature, u_ref.
    # fe_coll = mfem.H1_FECollection(order, dim)
    # fespace = mfem.FiniteElementSpace(mesh, fe_coll)

    # fe_size = fespace.GetTrueVSize()
    # print("Number of temperature unknowns: " + str(fe_size))

    # u_gf = mfem.GridFunction(fespace)
    # dudt_gf = mfem.GridFunction(fespace)

    # # 6. Set the initial conditions for u. All boundaries are considered
    # #    natural.
    # u_0 = InitialSolution()
    # dudt_0 = InitialRate()

    # u_gf.ProjectCoefficient(u_0)
    # u = mfem.Vector()
    # u_gf.GetTrueDofs(u)

    # dudt_gf.ProjectCoefficient(dudt_0)
    # dudt = mfem.Vector()
    # dudt_gf.GetTrueDofs(dudt)

    # # 7. Initialize the conduction operator and the visualization.
    # ess_bdr = mfem.intArray()
    # if mesh.bdr_attributes.Size():
    #     ess_bdr.SetSize(mesh.bdr_attributes.Max())
    #     if (dirichlet):
    #         ess_bdr.Assign(1)
    #     else:
    #         ess_bdr.Assigne(0)

    # oper = WaveOperator(fespace, ess_bdr, speed)

    # u_gf.SetFromTrueDofs(u)

    # mesh.Print("wave_equation.mesh", 8)
    # output = io.StringIO()
    # output.precision = 8
    # u_gf.Save(output)
    # dudt_gf.Save(output)
    # fid = open("wave_equation-init.gf", 'w')
    # fid.write(output.getvalue())
    # fid.close()

    # if visit:
    #     visit_dc = mfem.VisItDataCollection("Wave_Equation", mesh)
    #     visit_dc.RegisterField("solution", u_gf)
    #     visit_dc.RegisterField("rate", dudt_gf)
    #     visit_dc.SetCycle(0)
    #     visit_dc.SetTime(0.0)
    #     visit_dc.Save()

    # if visualization:
    #     sout = mfem.socketstream("localhost", 19916)
    #     if not sout.good():
    #         print("Unable to connect to GLVis server at localhost:19916")
    #         visualization = False
    #         print("GLVis visualization disabled.")
    #     else:
    #         sout.precision(precision)
    #         sout << "solution\n" << mesh << dudt_gf
    #         sout << "pause\n"
    #         sout.flush()
    #         print(
    #             "GLVis visualization paused. Press space (in the GLVis window) to resume it.")
            
    # # 8. Perform time-integration (looping over the time iterations, ti, with a
    # #    time-step dt).
    # # mfem::StopWatch is not binded by pyMFEM.
    # fom_timer, dmd_training_timer, dmd_prediction_timer = StopWatch(), StopWatch(), StopWatch()
    # fom_timer.Start()
    # ode_solver.Init(oper)
    # t = 0.0
    # fom_timer.Stop()
    # dmd_training_timer.Start()
    # curr_window = 0
    # ts, dmd_u = [], []
    # dmd_u += [algo.DMD(u.Size(), dt)]

    # # NOTE: mfem Vector::GetData returns a SWIG Object of type double *.
    # # To make it compatible with pybind11, we use ctypes to read data from the memory address.
    # from ctypes import *
    # uData = (c_double * u.Size()).from_address(int(u.GetData())) # this does not copy the data.
    # # uData = list(uData) # this copies the data.
    # uData = np.array(uData, copy=False)

    # # Showing the memory address info
    # print("All of these memory addresses are different.")
    # print("id(uData[0]): %d" % id(uData[0]))
    # print("int(u.GetData()): %d" % (int(u.GetData()))) # this is not the same as u[0], yet still points to the data.
    # print("id(uData): %d" % id(uData))              # this is not the same as u[0], yet still points to the data.

    # print("But uData[*] points to the right memory.")
    # print("id(u[0]): %d =? id(uData[0]): %d" % (id(u[0]), id(uData[0])))
    # print("id(u[1]): %d =? id(uData[1]): %d" % (id(u[1]), id(uData[1])))
    # print("uData type: %s" % type(uData))

    # dmd_u[curr_window].takeSample(uData, t)
    # ts += [t]
    # dmd_training_timer.Stop()

    # last_step = False
    # ti = 0
    # while not last_step:
    #     ti += 1
    #     if t + dt >= t_final - dt/2:
    #         last_step = True

    #     fom_timer.Start()
    #     t, dt = ode_solver.Step(u, dudt, t, dt)
    #     fom_timer.Stop()

    #     dmd_training_timer.Start()
    #     dmd_u[curr_window].takeSample(uData, t)
        
    #     if (last_step or (ti % windowNumSamples == 0)):
    #         print("step %d, t= %f" % (ti, t))

    #         if (rdim != -1):
    #             print("Creating DMD with rdim %d at window index: %d" % (rdim, curr_window))
    #             dmd_u[curr_window].train(rdim)
    #         else:
    #             print("Creating DMD with energy fraction: %f at window index: %d" % (ef, curr_window))
    #             dmd_u[curr_window].train(ef)

    #         if (not last_step):
    #             curr_window += 1
    #             dmd_u += [algo.DMD(u.Size(), dt)]
    #             dmd_u[curr_window].takeSample(uData, t)
    #     ts += [t]
    #     dmd_training_timer.Stop()

    #     if last_step or (ti % vis_steps == 0):
    #         print("step " + str(ti) + ", t = " + "{:g}".format(t))

    #         u_gf.SetFromTrueDofs(u)
    #         dudt_gf.SetFromTrueDofs(dudt)
    #         if visualization:
    #             sout << "solution\n" << mesh << u_gf
    #             sout.flush()

    #         if visit:
    #             visit_dc.SetCycle(ti)
    #             visit_dc.SetTime(t)
    #             visit_dc.Save()

    #     oper.SetParameters(u)

    # # 9. Save the final solution. This output can be viewed later using GLVis:
    # #    "glvis -m wave_equation.mesh -g wave_equation-final.gf".
    # output = io.StringIO()
    # output.precision = 8
    # u_gf.Save(output)
    # dudt_gf.Save(output)
    # fid = open("wave_equation-final.gf", 'w')
    # fid.write(output.getvalue())
    # fid.close()

    # # 10. Predict the state at t_final using DMD.
    # print("Predicting temperature using DMD")
    # dmd_visit_dc = mfem.VisItDataCollection("DMD_Wave_Equation", mesh)
    # dmd_visit_dc.RegisterField("solution", u_gf)
    # curr_window = 0
    # if (visit):
    #     dmd_prediction_timer.Start()
    #     result_u = dmd_u[curr_window].predict(ts[0])
    #     dmd_prediction_timer.Stop()

    #     # result_u.getData() returns a numpy array, which shares the memory buffer.
    #     # result_u.getData() does not own the memory.
    #     initial_dmd_solution_u = mfem.Vector(result_u.getData(), result_u.dim())
    #     u_gf.SetFromTrueDofs(initial_dmd_solution_u)
    #     dmd_visit_dc.SetCycle(0)
    #     dmd_visit_dc.SetTime(0.0)
    #     dmd_visit_dc.Save()

    # for i in range(1, len(ts)):
    #     if ((i == len(ts) - 1) or (i % vis_steps == 0)):
    #         if (visit):
    #             dmd_prediction_timer.Start()
    #             result_u = dmd_u[curr_window].predict(ts[i])
    #             dmd_prediction_timer.Stop()

    #             dmd_solution_u = mfem.Vector(result_u.getData(), result_u.dim())
    #             u_gf.SetFromTrueDofs(dmd_solution_u)
    #             dmd_visit_dc.SetCycle(i)
    #             dmd_visit_dc.SetTime(ts[i])
    #             dmd_visit_dc.Save()

    #         if ((i % windowNumSamples == 0) and (i < len(ts)-1)):
    #             curr_window += 1

    # dmd_prediction_timer.Start()
    # result_u = dmd_u[curr_window].predict(t_final)
    # dmd_prediction_timer.Stop()

    # # 11. Calculate the relative error between the DMD final solution and the true solution.
    # dmd_solution_u = mfem.Vector(result_u.getData(), result_u.dim())
    
    # diff_u = mfem.Vector(u.Size())
    # mfem.subtract_vector(dmd_solution_u, u, diff_u)
    # tot_diff_norm_u = np.sqrt(mfem.InnerProduct(diff_u, diff_u))
    # tot_true_solution_u_norm = np.sqrt(mfem.InnerProduct(u, u))

    # print("Relative error of DMD solution (u) at t_final: %f is %.3E" % (t_final, tot_diff_norm_u / tot_true_solution_u_norm))
    # print("Elapsed time for solving FOM: %e second\n" % fom_timer.duration)
    # print("Elapsed time for training DMD: %e second\n" % dmd_training_timer.duration)
    # print("Elapsed time for predicting DMD: %e second\n" % dmd_prediction_timer.duration)

if __name__ == "__main__":
    run()