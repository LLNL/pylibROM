'''
   libROM MFEM Example: dg_advection_global_rom (adapted from ex9p.cpp)

      This example code solves the time-dependent advection equation
      du/dt + v.grad(u) = 0, where v is a given fluid velocity, and
      u0(x)=u(0,x) is a given initial condition.

      The example demonstrates the use of Discontinuous Galerkin (DG)
      bilinear forms in MFEM (face integrators), the use of implicit
      and explicit ODE time integrators, the definition of periodic
      boundary conditions through periodic meshes, as well as the use
      of GLVis for persistent visualization of a time-evolving
      solution. Saving of time-dependent data files for visualization
      with VisIt (visit.llnl.gov) and ParaView (paraview.org), as
      well as the optional saving with ADIOS2 (adios2.readthedocs.io)
      are also illustrated.

   How to run:
      mpirun -np 8 python <arguments>

      Arguments of reproductive case:
      dg_advection_global_rom.py -offline
      dg_advection_global_rom.py -merge -ns 1
      dg_advection_global_rom.py -online

      Outputs of reproductive case:
      Relative L2 error of ROM solution = 5.64184E-04

      Arguments of parametric predictive case:
      Offline phase: dg_advection_global_rom.py -offline -ff 1.0 -id 0
                     dg_advection_global_rom.py -offline -ff 1.1 -id 1
                     dg_advection_global_rom.py -offline -ff 1.2 -id 2

      Merge phase: dg_advection_global_rom.py -merge -ns 3

      FOM solution: dg_advection_global_rom.py -fom -ff 1.15

      Online phase: dg_advection_global_rom.py -online -ff 1.15

      Outputs of parametric predictive case:
      Relative L2 error of ROM solution = 4.33318E-04
'''

from mfem import path
import mfem.par as mfem
from mfem.par import intArray
from mfem.common.arg_parser import ArgParser
from os.path import expanduser, join, dirname, exists
from mpi4py import MPI
import numpy as np
from numpy import sqrt, pi, cos, sin, hypot, arctan2
from scipy.special import erfc

import sys
from ctypes import c_double
import pylibROM.linalg as libROM
from pylibROM.python_utils.StopWatch import StopWatch
from pylibROM.mfem import ComputeCtAB, ComputeCtAB_vec

num_proc = MPI.COMM_WORLD.size
myid = MPI.COMM_WORLD.rank

parser = ArgParser(description='dg_advection_global_rom')
parser.add_argument('-m', '--mesh',
                    default='periodic-hexagon.mesh',
                    action='store', type=str,
                    help="Mesh file to use.")
parser.add_argument('-p', '--problem',
                    action='store', default=3, type=int,
                    help="Problem setup to use")
parser.add_argument('-ff', '--f-factor',
                    action='store', default=1.0, type=float,
                    help="Frequency scalar factor")
parser.add_argument('-rs', '--refine-serial',
                    action='store', default=2, type=int,
                    help="Number of times to refine the mesh uniformly before parallel")
parser.add_argument('-rp', '--refine-parallel',
                    action='store', default=0, type=int,
                    help="Number of times to refine the mesh uniformly after parallel")
parser.add_argument('-o', '--order',
                    action='store', default=3, type=int,
                    help="Finite element order (polynomial degree)")
help_ode = "\n".join(["ODE solver: 1 - Forward Euler",
                      "\t2 - RK2 SSP, 3 - RK3 SSP, 4 - RK4, 6 - RK6",
                      "\t11 - Backward Euler,",
                      "\t12 - SDIRK23 (L-stable), 13 - SDIRK33,",
                      "\t22 - Implicit Midpoint Method,",
                      "\t23 - SDIRK23 (A-stable), 24 - SDIRK34"])
parser.add_argument('-s', '--ode-solver-type',
                    action='store', default=11, type=int,
                    help=help_ode)
parser.add_argument('-tf', '--t-final',
                    action='store', default=10.0, type=float,
                    help="Final time; start time is 0.")
parser.add_argument('-dt', '--time-step',
                    action='store', default=0.01, type=float,
                    help="Time step")
parser.add_argument("-vs", "--visualization-steps",
                    action='store', default=5, type=int,
                    help="Visualize every n-th timestep.")
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
parser.add_argument('-ef','--energy_fraction',
                    action='store', default=0.9999, type=float,
                    help="Energy fraction for POD")
parser.add_argument('-rdim','--rdim',
                    action='store', default=-1, type=int,
                    help="Reduced dimension for POD")
parser.add_argument("-id", "--id",
                    action='store', default=0, type=int, help="Parametric id")
parser.add_argument("-ns", "--nsets",
                    action='store', default=1, type=int, help="Number of parametric snapshot sets")
args = parser.parse_args()

problem = args.problem
f_factor = args.f_factor
ser_ref_levels = args.refine_serial
par_ref_levels = args.refine_parallel
order = args.order
ode_solver_type = args.ode_solver_type
t_final = args.t_final
dt = args.time_step
vis_steps = args.visualization_steps
fom = args.fom
offline = args.offline
online = args.online
merge = args.merge
ef = args.energy_fraction
rdim = args.rdim
id = args.id
nsets = args.nsets

if fom:
    if (not (fom and (not offline) and (not online))):
        raise ValueError("offline and online must be turned off if fom is used.")
else:
    check = (offline and (not merge) and (not online))          \
            or ((not offline) and merge and (not online))       \
            or ((not offline) and (not merge) and online)
    if (not check):
        raise ValueError("only one of offline, merge, or online must be true!")

device = mfem.Device('cpu')
if myid == 0:
    device.Print()

# initialize timers
solveTimer, assembleTimer, mergeTimer = \
        StopWatch(), StopWatch(), StopWatch()

# 3. Read the serial mesh from the given mesh file on all processors. We can
#    handle geometrically periodic meshes in this code.
meshfile = expanduser(join(dirname(__file__), '..', 'data', args.mesh))
if not exists(meshfile):
    path = dirname(dirname(__file__))
    meshfile = expanduser(join(dirname(__file__), '..', 'data', args.mesh))

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
            return sin(f_factor * pi * X[0]) * sin(f_factor * pi * X[1])

        return 0.0

# Inflow boundary condition (zero for the problems considered in this example)


class inflow_coeff(mfem.PyCoefficient):
    def EvalValue(self, x):
        return 0


# 8. Set up and assemble the bilinear and linear forms corresponding to the
#    DG discretization. The DGTraceIntegrator involves integrals over mesh
#    interior faces.

assembleTimer.Start()
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
assembleTimer.Stop()

# 9. Define the initial conditions, save the corresponding grid function to
#    a file
u = mfem.ParGridFunction(fes)
u.ProjectCoefficient(u0)
U = u.GetTrueDofs()
u_init = mfem.Vector(U)

smyid = '{:0>6d}'.format(myid)
mesh_name = "ex9-mesh."+smyid
sol_name = "ex9-init."+smyid
pmesh.Print(mesh_name, 8)
u.Save(sol_name, 8)

# 10. Evolution operator
class FE_Evolution(mfem.PyTimeDependentOperator):
    def __init__(self, M, K, b):
        mfem.PyTimeDependentOperator.__init__(self, M.Height())

        self.M_prec = mfem.HypreSmoother()
        self.M_solver = mfem.CGSolver(M.GetComm())
        self.T_prec = mfem.HypreSmoother()
        self.T_solver = mfem.CGSolver(M.GetComm())
        self.z = mfem.Vector(M.Height())

        self.K = K
        self.M = M
        self.T = None
        self.b = b
        self.M_prec.SetType(mfem.HypreSmoother.Jacobi)
        self.M_solver.SetPreconditioner(self.M_prec)
        self.M_solver.SetOperator(M)
        self.M_solver.iterative_mode = False
        self.M_solver.SetRelTol(1e-9)
        self.M_solver.SetAbsTol(0.0)
        self.M_solver.SetMaxIter(1000)
        self.M_solver.SetPrintLevel(0)
        self.T_solver.SetPreconditioner(self.T_prec)
        self.T_solver.iterative_mode = False
        self.T_solver.SetRelTol(1e-9)
        self.T_solver.SetAbsTol(0.0)
        self.T_solver.SetMaxIter(1000)
        self.T_solver.SetPrintLevel(0)

    def Mult(self, x, y):
        # M y = K x + b
        self.K.Mult(x, self.z)
        self.z += self.b
        self.M_solver.Mult(self.z, y)

    def ImplicitSolve(self, dt, x, k):
        # (M - dt*K) k = K x + b
        if self.T is None:
            self.T = mfem.Add(1.0, self.M, -dt, self.K)
            current_dt = dt
            self.T_solver.SetOperator(self.T)
        self.K.Mult(x, self.z)
        self.z += self.b
        self.T_solver.Mult(self.z, k)

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
        if self.Ainv is None:
            #self.Ainv = mfem.DenseMatrix(self.M.NumRows(), self.M.NumCols())
            #self.Ainv.Assign(0)
            #self.Ainv.Add(1.0, self.M)
            #self.Ainv.Add(-dt, self.K)
            A = mfem.DenseMatrix(self.K.NumRows(), self.K.NumCols())
            A.Set(dt, self.K)
            self.Ainv = mfem.DenseMatrix(self.M)
            self.Ainv -= A
            current_dt = dt
            self.Ainv.Invert()
        self.K.Mult(x, self.z)
        self.z += self.b
        self.z += self.u_init_hat
        self.Ainv.Mult(self.z, k)

# 11. Initiate ROM related variables
# ROM object options
max_num_snapshots = 100000
update_right_SV = False
isIncremental = False
basisName = "basis"
basisFileName = "%s%d" % (basisName, id)

# BasisGenerator for snapshot collection in offline phase 
if offline:
    options = libROM.Options(fes.GetTrueVSize(), max_num_snapshots,
                            update_right_SV)
    generator = libROM.BasisGenerator(options, isIncremental, basisFileName)

# BasisGenerator for basis construction in online phase 
if merge:
    mergeTimer.Start()
    options = libROM.Options(fes.GetTrueVSize(), max_num_snapshots,
                            update_right_SV)
    generator = libROM.BasisGenerator(options, isIncremental, basisName)
    for paramID in range(nsets):
        snapshot_filename = "%s%d_snapshot" % (basisName, paramID)
        generator.loadSamples(snapshot_filename,"snapshot") # this is much slower than C++

    generator.endSamples() # save the merged basis file
    mergeTimer.Stop()
    if myid == 0:
        print("Elapsed time for merging and building ROM basis: %e second\n" %
               mergeTimer.duration)
    del generator
    del options
    MPI.Finalize()
    sys.exit(0)

# 12. Assemble evolution operator
assembleTimer.Start()
if online:
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
    ComputeCtAB(M, spatialbasis, spatialbasis, M_hat_carom)
    M_hat = mfem.DenseMatrix(M_hat_carom.getData())
    # Unlike C++, the conversion from libROM.Matrix to mfem.DenseMatrix does not need tranpose
    # M_hat.Transpose()

    K_hat_carom = libROM.Matrix(numColumnRB, numColumnRB, False)
    ComputeCtAB(K, spatialbasis, spatialbasis, K_hat_carom)
    K_hat = mfem.DenseMatrix(K_hat_carom.getData())
    # Unlike C++, the conversion from libROM.Matrix to mfem.DenseMatrix does not need tranpose
    # K_hat.Transpose()

    b_vec = np.array((c_double * B.Size()).from_address(int(B.GetData())), copy=False)
    b_carom = libROM.Vector(b_vec, True)
    b_hat_carom = spatialbasis.transposeMult(b_carom)
    b_hat = mfem.Vector(b_hat_carom.getData(), b_hat_carom.dim())

    u_init_hat_carom = libROM.Vector(numColumnRB, False)
    ComputeCtAB_vec(K, U, spatialbasis, u_init_hat_carom)
    u_init_hat = mfem.Vector(u_init_hat_carom.getData(), u_init_hat_carom.dim())

    u_hat = mfem.Vector(numColumnRB)
    u_hat.Assign(0.0)

    adv = ROM_FE_Evolution(M_hat, K_hat, b_hat, u_init_hat, numColumnRB)
else:
    adv = FE_Evolution(M, K, B)
adv.SetTime(0.0)
ode_solver.Init(adv)
assembleTimer.Stop()

# 13. Time marching
t = 0.0
ti = 0
done = False
if offline:
    u_curr = mfem.Vector(U)
    u_centered = mfem.Vector(U.Size())
    mfem.subtract_vector(u_curr, u_init, u_centered);
    u_centered_vec = np.array((c_double * U.Size()).from_address(int(u_centered.GetData())), copy=False)
    addSample = generator.takeSample(u_centered_vec)

while not done:
    dt_real = min(dt, t_final - t)
    solveTimer.Start()
    if online:
        t, dt = ode_solver.Step(u_hat, t, dt_real)
    else:
        t, dt = ode_solver.Step(U, t, dt_real)
    solveTimer.Stop()
    ti = ti + 1

    if t >= t_final - 1e-8 * dt:
        done = True

    if offline:
        u_curr = mfem.Vector(U)
        u_centered = mfem.Vector(U.Size())
        mfem.subtract_vector(u_curr, u_init, u_centered);
        u_centered_vec = np.array((c_double * u_centered.Size()).from_address(int(u_centered.GetData())), copy=False)
        addSample = generator.takeSample(u_centered_vec)

    if done or ti % vis_steps == 0:
        if myid == 0:
            print("time step: " + str(ti) + ", time: " + str(np.round(t, 3)))

# 14. Compute basis
if offline:
    generator.writeSnapshot()
    del generator
    del options

solution_filename_fom = "dg_advection_global_rom-final.%06d" % f_factor
# 15. Save and compare solution
if online:
    u_hat_final_vec = np.array((c_double * u_hat.Size()).from_address(int(u_hat.GetData())), copy=False)
    u_hat_final_carom = libROM.Vector(u_hat_final_vec, False)
    u_final_carom = libROM.Vector(U.Size(), True)
    spatialbasis.mult(u_hat_final_carom, u_final_carom)
    u_final = mfem.Vector(u_final_carom.getData(), u_final_carom.dim())
    u_final += u_init
    fom_solution = mfem.Vector(u_final.Size())
    fom_solution.Load(solution_filename_fom, u_final.Size())
    fomNorm = np.sqrt(mfem.InnerProduct(MPI.COMM_WORLD, fom_solution, fom_solution))
    diff_solution = mfem.Vector(u_final.Size())
    mfem.subtract_vector(fom_solution, u_final, diff_solution) 
    diffNorm = np.sqrt(mfem.InnerProduct(MPI.COMM_WORLD, diff_solution, diff_solution))
    if myid == 0:
        print("Relative L2 error of ROM solution = %.5E" % (diffNorm / fomNorm))
        print("Elapsed time for assembling ROM: %e second\n" % assembleTimer.duration)
        print("Elapsed time for solving ROM: %e second\n" % solveTimer.duration)

if offline or fom:
    u = np.array((c_double * U.Size()).from_address(int(U.GetData())), copy=False)
    np.savetxt(solution_filename_fom, u, fmt='%.16f')
    if myid == 0:
        print("Elapsed time for assembling FOM: %e second\n" % assembleTimer.duration)
        print("Elapsed time for solving FOM: %e second\n" % solveTimer.duration)

del fes
MPI.Finalize()
