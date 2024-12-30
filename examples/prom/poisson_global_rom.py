'''
//               pylibROM PyMFEM Example: parametric ROM for Poisson problem (adapted from ex1p.cpp)
//
// Description:  This example code demonstrates the use of PyMFEM and pylibROM to
//               define a simple projection-based reduced order model of the
//               Poisson problem -Delta u = f(x) with homogeneous Dirichlet
//               boundary conditions and spatially varying right hand side f.
//
//               The example highlights three distinct ROM processes, i.e.,
//               offline, merge, and online. The offline phase runs the full
//               order model and stores the snapshot data in an HDF file. You
//               can run as many offline phases as you wish to sample the
//               parameter space. The merge phase reads all the snapshot files,
//               builds a global reduced basis, and stores the basis in an HDF
//               file. The online phase reads the basis, builds the ROM
//               operator, solves the reduced order system, and lifts the
//               solution to the full order space.
//
// Offline phase: python3 poisson_global_rom.py -offline -f 1.0 -id 0
//                python3 poisson_global_rom.py -offline -f 1.1 -id 1
//                python3 poisson_global_rom.py -offline -f 1.2 -id 2
//
// Merge phase:   python3 poisson_global_rom.py -merge -ns 3
//
// FOM run for error calculation:
//                python3 poisson_global_rom.py -fom -f 1.15
//
// Online phase:  python3 poisson_global_rom.py -online -f 1.15
//
// This example runs in parallel with MPI, by using the same number of MPI ranks
// in all phases (offline, merge, fom, online).
'''
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
from mfem.par import intArray
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
from pylibROM.mfem import ComputeCtAB

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
    if (myid == 0): parser.print_options(args)

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
    visit           = args.visit_datafiles
    visualization   = args.visualization

    precision       = 8

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
    pmesh = mfem.ParMesh(comm, mesh)
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

    fespace = mfem.ParFiniteElementSpace(pmesh, fec)
    size = fespace.GlobalTrueVSize()
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
    solveTimer, assembleTimer, mergeTimer = StopWatch(), StopWatch(), StopWatch()

    # 10. Set BasisGenerator if offline
    if (offline):
        options = libROM.Options(fespace.GetTrueVSize(), max_num_snapshots,
                                 update_right_SV)
        generator = libROM.BasisGenerator(options, isIncremental, basisFileName)

    # 11. The merge phase
    if (merge):
        mergeTimer.Start()
        options = libROM.Options(fespace.GetTrueVSize(), max_num_snapshots,
                                 update_right_SV)
        generator = libROM.BasisGenerator(options, isIncremental, basisName)
        for paramID in range(nsets):
            snapshot_filename = "%s%d_snapshot" % (basisName, paramID)
            generator.loadSamples(snapshot_filename,"snapshot", 5)

        generator.endSamples() # save the merged basis file
        mergeTimer.Stop()
        if (myid == 0):
            print("Elapsed time for merging and building ROM basis: %e second\n" %
                   mergeTimer.duration)
        del generator
        del options
        MPI.Finalize()
        return

    # 12. Set up the parallel linear form b(.) which corresponds to the
    #     right-hand side of the FEM linear system, which in this case is
    #     (f,phi_i) where f is given by the function f_exact and phi_i are the
    #     basis functions in the finite element fespace.
    assembleTimer.Start()
    b = mfem.ParLinearForm(fespace)
    class RightHandSide(mfem.PyCoefficient):
        def EvalValue(self, x):
            if (dim == 3):
                return sin(kappa * (x[0] + x[1] + x[2]))
            else:
                return sin(kappa * (x[0] + x[1]))
    f = RightHandSide()
    b.AddDomainIntegrator(mfem.DomainLFIntegrator(f))
    b.Assemble()

    # 13. Define the solution vector x as a parallel finite element grid function
    #     corresponding to fespace. Initialize x with initial guess of zero,
    #     which satisfies the boundary conditions.
    x = mfem.ParGridFunction(fespace)
    x.Assign(0.0)

    # 14. Set up the parallel bilinear form a(.,.) on the finite element space
    #     corresponding to the Laplacian operator -Delta, by adding the Diffusion
    #     domain integrator.
    a = mfem.ParBilinearForm(fespace)
    one = mfem.ConstantCoefficient(coef)
    if (pa):
        a.SetAssemblyLevel(mfem.AssemblyLevel_PARTIAL)
    a.AddDomainIntegrator(mfem.DiffusionIntegrator(one))

    # 15. Assemble the parallel bilinear form and the corresponding linear
    #     system, applying any necessary transformations such as: parallel
    #     assembly, eliminating boundary conditions, applying conforming
    #     constraints for non-conforming AMR, static condensation, etc.
    if (static_cond):
        a.EnableStaticCondensation()
    a.Assemble()

    # A = mfem.OperatorPtr()
    A = mfem.HypreParMatrix()
    B = mfem.Vector()
    X = mfem.Vector()
    a.FormLinearSystem(ess_tdof_list, x, b, A, X, B)
    assembleTimer.Stop()

    # 16. The offline phase
    if (fom or offline):
        # 17. Solve the full order linear system A X = B
        prec = None
        if pa:
            if mfem.UsesTensorBasis(fespace):
                prec = mfem.OperatorJacobiSmoother(a, ess_tdof_list)
        else:
            prec = mfem.HypreBoomerAMG(A)

        cg = mfem.CGSolver(comm)
        cg.SetRelTol(1e-12)
        cg.SetMaxIter(2000)
        cg.SetPrintLevel(1)
        if (prec is not None):
            cg.SetPreconditioner(prec)
        # cg.SetOperator(A.Ptr())
        cg.SetOperator(A)
        solveTimer.Start()
        cg.Mult(B, X)
        solveTimer.Stop()
        if (prec is not None):
            del prec

        # 18. take and write snapshot for ROM
        if (offline):
            # NOTE: mfem Vector::GetData returns a SWIG Object of type double *.
            # To make it compatible with pybind11, we use ctypes to read data from the memory address.
            xData = np.array((c_double * X.Size()).from_address(int(X.GetData())), copy=False) # this does not copy the data.
            addSample = generator.takeSample(xData)
            generator.writeSnapshot()
            del generator
            del options

    # 19. The online phase
    if (online):
        # 20. read the reduced basis
        assembleTimer.Start()
        reader = libROM.BasisReader(basisName)
        spatialbasis = reader.getSpatialBasis()
        numRowRB = spatialbasis.numRows()
        numColumnRB = spatialbasis.numColumns()
        if (myid == 0):
            print("spatial basis dimension is %d x %d\n" % (numRowRB, numColumnRB))

        # 21. form inverse ROM operator
        invReducedA = libROM.Matrix(numColumnRB, numColumnRB, False)
        ComputeCtAB(A, spatialbasis, spatialbasis, invReducedA)
        invReducedA.invert()

        bData = np.array((c_double * B.Size()).from_address(int(B.GetData())), copy=False)
        B_carom = libROM.Vector(bData, True, False)
        xData = np.array((c_double * X.Size()).from_address(int(X.GetData())), copy=False)
        X_carom = libROM.Vector(xData, True, False)
        reducedRHS = spatialbasis.transposeMult(B_carom)
        reducedSol = libROM.Vector(numColumnRB, False)
        assembleTimer.Stop()

        # 22. solve ROM
        solveTimer.Start()
        invReducedA.mult(reducedRHS, reducedSol)
        solveTimer.Stop()

        # 23. reconstruct FOM state
        spatialbasis.mult(reducedSol, X_carom)
        del spatialbasis
        del reducedRHS

    # 24. Recover the parallel grid function corresponding to X. This is the
    #     local finite element solution on each processor.
    a.RecoverFEMSolution(X, b, x)

    # 25. Calculate the relative error of the ROM prediction compared to FOM
    # ostringstream sol_dofs_name, sol_dofs_name_fom;
    if (fom or offline):
        sol_dofs_name = "sol_dofs_fom.%06d" % myid
    if (online):
        sol_dofs_name = "sol_dofs.%06d" % myid
        sol_dofs_name_fom = "sol_dofs_fom.%06d" % myid

    if (online):
        # Initialize FOM solution
        x_fom = mfem.Vector(x.Size())

        # Open and load file
        x_fom.Load(sol_dofs_name_fom, x_fom.Size())

        diff_x = mfem.Vector(x.Size())

        mfem.subtract_vector(x, x_fom, diff_x)

        # Get norms
        tot_diff_norm = np.sqrt(mfem.InnerProduct(comm, diff_x, diff_x))
        tot_fom_norm = np.sqrt(mfem.InnerProduct(comm, x_fom, x_fom))

        if (myid == 0):
            print("Relative error of ROM solution = %.5E" % (tot_diff_norm / tot_fom_norm))

    # 26. Save the refined mesh and the solution in parallel. This output can
    #     be viewed later using GLVis: "glvis -np <np> -m mesh -g sol".
    mesh_name  = "mesh.%06d" % myid
    sol_name = "sol.%06d" % myid

    pmesh.Print(mesh_name, precision)

    output = io.StringIO()
    output.precision = precision
    x.Save(output)
    fid = open(sol_name, 'w')
    fid.write(output.getvalue())
    fid.close()

    xData = np.array((c_double * X.Size()).from_address(int(X.GetData())), copy=False)
    np.savetxt(sol_dofs_name, xData, fmt='%.16f')

    # 27. Save data in the VisIt format.
    if (visit):
        if (offline):
            dc = mfem.VisItDataCollection("Example1", pmesh)
        elif (fom):
            dc = mfem.VisItDataCollection("Example1_fom", pmesh)
        elif (online):
            dc = mfem.VisItDataCollection("Example1_rom", pmesh)
        dc.SetPrecision(precision)
        dc.RegisterField("solution", x)
        dc.Save()
        del dc

    # 28. Send the solution by socket to a GLVis server.
    if visualization:
        sol_sock = mfem.socketstream("localhost", 19916)
        if not sol_sock.good():
            visualization = False
            if (myid == 0):
                print("Unable to connect to GLVis server at localhost:19916")
                print("GLVis visualization disabled.")
        else:
            sol_sock << "parallel " << num_procs << " " << myid << "\n"
            sol_sock.precision(precision)
            sol_sock << "solution\n" << pmesh << x
            sol_sock << "pause\n"
            sol_sock.flush()
            if (myid == 0):
                print(
                    "GLVis visualization paused. Press space (in the GLVis window) to resume it.")

    # 29. print timing info
    if (myid == 0):
        if (fom or offline):
            print("Elapsed time for assembling FOM: %e second\n" % assembleTimer.duration)
            print("Elapsed time for solving FOM: %e second\n" % solveTimer.duration)

        if(online):
            print("Elapsed time for assembling ROM: %e second\n" % assembleTimer.duration)
            print("Elapsed time for solving ROM: %e second\n" % solveTimer.duration)

    # 30. Free the used memory.
    if (delete_fec):
        del fec
    MPI.Finalize()

if __name__ == "__main__":
    run()
