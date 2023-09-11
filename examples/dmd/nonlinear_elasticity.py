'''
   libROM MFEM Example: nonlinear_elasticity (adapted from ex10p.cpp)


      This examples solves a time dependent nonlinear elasticity
      problem of the form dv/dt = H(x) + S v, dx/dt = v, where H is a
      hyperelastic model and S is a viscosity operator of Laplacian
      type.

   How to run:
      mpirun -np 8 python <arguments>

      Example 1 of arguments:
      nonlinear_elasticity.py -s 2 -rs 1 -dt 0.01 -tf 5 -visit
   
      Output 1:
      Relative error of DMD velocity (v) at t_final: 5.000000 is 1.145E-03
      Relative error of DMD position (x) at t_final: 5.000000 is 7.371E-05

      Example 2 of arguments:
      nonlinear_elasticity.py -s 2 -rs 1 -dt 0.01 -tf 5 -nwinsamp 10 -visit
   
      Output 2:
      Relative error of DMD velocity (v) at t_final: 5.000000 is 1.106E-04
      Relative error of DMD position (x) at t_final: 5.000000 is 4.304E-06
'''
import sys
import mfem.par as mfem

from mfem.common.arg_parser import ArgParser
from os.path import expanduser, join, dirname
import numpy as np
from numpy import sqrt, pi, cos, sin, hypot, arctan2
from scipy.special import erfc

from pylibROM.python_utils.StopWatch import StopWatch
from pylibROM.algo import DMD

from mfem.par import intArray, add_vector
from mpi4py import MPI

num_procs = MPI.COMM_WORLD.size
myid = MPI.COMM_WORLD.rank

parser = ArgParser(description='nonlinear_elasticity')
parser.add_argument('-m', '--mesh',
                    default='beam-quad.mesh',
                    action='store', type=str,
                    help='Mesh file to use.')
parser.add_argument('-rs', '--refine-serial',
                    action='store', default=2, type=int,
                    help="Number of times to refine the mesh uniformly before parallel")
parser.add_argument('-rp', '--refine-parallel',
                    action='store', default=0, type=int,
                    help="Number of times to refine the mesh uniformly after parallel")
parser.add_argument('-o', '--order',
                    action='store', default=2, type=int,
                    help="Finite element order (polynomial degree)")
help_ode = "\n".join(["ODE solver: 1 - Backward Euler, 2 - SDIRK2, 3 - SDIRK3",
                      "\t11 - Forward Euler, 12 - RK2",
                      "\t13 - RK3 SSP, 14 - RK4."])
parser.add_argument('-s', '--ode-solver',
                    action='store', default=3, type=int,
                    help=help_ode)
parser.add_argument('-tf', '--t-final',
                    action='store', default=300.0, type=float,
                    help="Final time; start time is 0.")
parser.add_argument('-dt', '--time-step',
                    action='store', default=3.0, type=float,
                    help="Time step")
parser.add_argument("-v", "--viscosity",
                    action='store', default=1e-2, type=float,
                    help="Viscosity coefficient.")
parser.add_argument("-mu", "--shear-modulus",
                    action='store', default=0.25, type=float,
                    help="Shear modulus in the Neo-Hookean hyperelastic model.")
parser.add_argument("-K", "--bulk-modulus",
                    action='store', default=5.0, type=float,
                    help="Bulk modulus in the Neo-Hookean hyperelastic model.")
parser.add_argument('-vis', '--visualization',
                    action='store_true', default=True,
                    help='Enable GLVis visualization')
parser.add_argument('-visit', '--visit-datafiles',
                    action='store_true', default=False,
                    help="Save data files for VisIt (visit.llnl.gov) visualization.")
parser.add_argument("-vs", "--visualization-steps",
                    action='store', default=1, type=int,
                    help="Visualize every n-th timestep.")
parser.add_argument('-ef','--energy_fraction',
                    action='store', default=0.9999, type=float,
                    help='Energy fraction for DMD')
parser.add_argument('-rdim','--rdim',
                    action='store', default=-1, type=int,
                    help='Reduced dimension for DMD')
parser.add_argument('-nwinsamp','--numwindowsamples',
                    action='store', default=sys.maxsize, type=int,
                    help='Number of samples in DMD windows.');

args = parser.parse_args()

ser_ref_levels = args.refine_serial
par_ref_levels = args.refine_parallel
order = args.order
ode_solver_type = args.ode_solver
t_final = args.t_final
dt = args.time_step
visc = args.viscosity
mu = args.shear_modulus
K = args.bulk_modulus
visualization = args.visualization
visit = args.visit_datafiles
vis_steps = args.visualization_steps
ef = args.energy_fraction
rdim = args.rdim
windowNumSamples = args.numwindowsamples

if (myid == 0):
    parser.print_options(args)

device = mfem.Device('cpu')
if myid == 0:
    device.Print()

# 3. Read the serial mesh from the given mesh file on all processors. We can
#    handle triangular, quadrilateral, tetrahedral and hexahedral meshes
#    with the same code.
meshfile = expanduser(join(dirname(__file__), '..', 'data', args.mesh))
mesh = mfem.Mesh(meshfile, 1, 1)
dim = mesh.Dimension()

# 4. Define the ODE solver used for time integration. Several implicit
#    singly diagonal implicit Runge-Kutta (SDIRK) methods, as well as
#    explicit Runge-Kutta methods are available.
if ode_solver_type == 1:
    ode_solver = BackwardEulerSolver()
elif ode_solver_type == 2:
    ode_solver = mfem.SDIRK23Solver(2)
elif ode_solver_type == 3:
    ode_solver = mfem.SDIRK33Solver()
elif ode_solver_type == 11:
    ode_solver = ForwardEulerSolver()
elif ode_solver_type == 12:
    ode_solver = mfem.RK2Solver(0.5)
elif ode_solver_type == 13:
    ode_solver = mfem.RK3SSPSolver()
elif ode_solver_type == 14:
    ode_solver = mfem.RK4Solver()
elif ode_solver_type == 22:
    ode_solver = mfem.ImplicitMidpointSolver()
elif ode_solver_type == 23:
    ode_solver = mfem.SDIRK23Solver()
elif ode_solver_type == 24:
    ode_solver = mfem.SDIRK34Solver()
else:
    if myid == 0:
        print("Unknown ODE solver type: " + str(ode_solver_type))
    sys.exit()

# 5. Refine the mesh in serial to increase the resolution. In this example
#    we do 'ser_ref_levels' of uniform refinement, where 'ser_ref_levels' is
#    a command-line parameter.
for lev in range(ser_ref_levels):
    mesh.UniformRefinement()

# 6. Define a parallel mesh by a partitioning of the serial mesh. Refine
#    this mesh further in parallel to increase the resolution. Once the
#    parallel mesh is defined, the serial mesh can be deleted.
pmesh = mfem.ParMesh(MPI.COMM_WORLD, mesh)
del mesh
for x in range(par_ref_levels):
    pmesh.UniformRefinement()

# 7. Define the parallel vector finite element spaces representing the mesh
#    deformation x_gf, the velocity v_gf, and the initial configuration,
#    x_ref. Define also the elastic energy density, w_gf, which is in a
#    discontinuous higher-order space. Since x and v are integrated in time
#    as a system, we group them together in block vector vx, on the unique
#    parallel degrees of freedom, with offsets given by array true_offset.

fec = mfem.H1_FECollection(order, dim)
fespace = mfem.ParFiniteElementSpace(pmesh, fec, dim)
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

x_ref = mfem.ParGridFunction(fespace)
pmesh.GetNodes(x_ref)

w_fec = mfem.L2_FECollection(order + 1, dim)
w_fespace = mfem.ParFiniteElementSpace(pmesh, w_fec)
w_gf = mfem.ParGridFunction(w_fespace)

# 8. Set the initial conditions for v_gf, x_gf and vx, and define the
#    boundary conditions on a beam-like mesh (see description above).


class InitialVelocity(mfem.VectorPyCoefficient):
    def EvalValue(self, x):
        dim = len(x)
        s = 0.1/64.

        v = np.zeros(len(x))
        v[-1] = s*x[0]**2*(8.0-x[0])
        v[0] = -s*x[0]**2
        return v


class InitialDeformation(mfem.VectorPyCoefficient):
    def EvalValue(self, x):
        return x.copy()


velo = InitialVelocity(dim)
v_gf.ProjectCoefficient(velo)
deform = InitialDeformation(dim)
x_gf.ProjectCoefficient(deform)

v_gf.GetTrueDofs(vx.GetBlock(0))
x_gf.GetTrueDofs(vx.GetBlock(1))

ess_bdr = mfem.intArray(fespace.GetMesh().bdr_attributes.Max())
ess_bdr.Assign(0)
ess_bdr[0] = 1


# 9. Initialize the hyperelastic operator, the GLVis visualization and print
#    the initial energies.
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


class ReducedSystemOperator(mfem.PyOperator):
    def __init__(self, M, S, H, ess_tdof_list):
        mfem.PyOperator.__init__(self, M.ParFESpace().TrueVSize())
        self.M = M
        self.S = S
        self.H = H
        self.Jacobian = None
        h = M.ParFESpace().TrueVSize()
        self.w = mfem.Vector(h)
        self.z = mfem.Vector(h)
        self.dt = 0.0
        self.ess_tdof_list = ess_tdof_list

    def SetParameters(self, dt, v, x):
        self.dt = dt
        self.v = v
        self.x = x

    def Mult(self, k, y):
        add_vector(self.v, self.dt, k, self.w)
        add_vector(self.x, self.dt, self.w, self.z)
        self.H.Mult(self.z, y)
        self.M.TrueAddMult(k, y)
        self.S.TrueAddMult(self.w, y)
        y.SetSubVector(self.ess_tdof_list, 0.0)

    def GetGradient(self, k):
        localJ = mfem.Add(1.0, self.M.SpMat(), self.dt, self.S.SpMat())
        add_vector(self.v, self.dt, k, self.w)
        add_vector(self.x, self.dt, self.w, self.z)
        localJ.Add(self.dt * self.dt,  self.H.GetLocalGradient(self.z))
        Jacobian = self.M.ParallelAssemble(localJ)
        Jacobian.EliminateRowsCols(self.ess_tdof_list)
        return Jacobian


class HyperelasticOperator(mfem.PyTimeDependentOperator):
    def __init__(self, fespace, ess_bdr, visc, mu, K):
        mfem.PyTimeDependentOperator.__init__(self, 2*fespace.TrueVSize(), 0.0)

        rel_tol = 1e-8
        skip_zero_entries = 0
        ref_density = 1.0

        self.ess_tdof_list = intArray()
        self.z = mfem.Vector(self.Height()//2)
        self.fespace = fespace
        self.viscosity = visc
        self.newton_solver = mfem.NewtonSolver(fespace.GetComm())

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

        fespace.GetEssentialTrueDofs(ess_bdr, self.ess_tdof_list)
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

        self.reduced_oper = ReducedSystemOperator(M, S, H, self.ess_tdof_list)

        J_hypreSmoother = mfem.HypreSmoother()
        J_hypreSmoother.SetType(mfem.HypreSmoother.l1Jacobi)
        J_hypreSmoother.SetPositiveDiagonal(True)
        J_prec = J_hypreSmoother

        J_minres = mfem.MINRESSolver(fespace.GetComm())
        J_minres.SetRelTol(rel_tol)
        J_minres.SetAbsTol(0.0)
        J_minres.SetMaxIter(300)
        J_minres.SetPrintLevel(-1)
        J_minres.SetPreconditioner(J_prec)

        self.J_solver = J_minres
        self.J_prec = J_prec

        newton_solver = mfem.NewtonSolver(fespace.GetComm())
        newton_solver.iterative_mode = False
        newton_solver.SetSolver(self.J_solver)
        newton_solver.SetOperator(self.reduced_oper)
        newton_solver.SetPrintLevel(1)  # print Newton iterations
        newton_solver.SetRelTol(rel_tol)
        newton_solver.SetAbsTol(0.0)
        newton_solver.SetAdaptiveLinRtol(2, 0.5, 0.9)
        newton_solver.SetMaxIter(10)
        self.newton_solver = newton_solver

    def Mult(self, vx, vx_dt):
        sc = self.Height()//2
        v = mfem.Vector(vx, 0,  sc)
        x = mfem.Vector(vx, sc,  sc)
        dv_dt = mfem.Vector(dvx_dt, 0, sc)
        dx_dt = mfem.Vector(dvx_dt, sc,  sc)

        self.H.Mult(x, z)
        if (self.viscosity != 0.0):
            S.TrueAddMult(v, z)
            z.SetSubVector(self.ess_tdof_list, 0.0)
        z.Neg()
        self.M_solver.Mult(z, dv_dt)
        dx_dt = v

    def ImplicitSolve(self, dt, vx, dvx_dt):
        sc = self.Height()//2
        v = mfem.Vector(vx, 0,  sc)
        x = mfem.Vector(vx, sc,  sc)
        dv_dt = mfem.Vector(dvx_dt, 0, sc)
        dx_dt = mfem.Vector(dvx_dt, sc,  sc)

        # By eliminating kx from the coupled system:
        # kv = -M^{-1}*[H(x + dt*kx) + S*(v + dt*kv)]
        # kx = v + dt*kv
        # we reduce it to a nonlinear equation for kv, represented by the
        # backward_euler_oper. This equation is solved with the newton_solver
        # object (using J_solver and J_prec internally).
        self.reduced_oper.SetParameters(dt, v, x)
        zero = mfem.Vector()  # empty vector is interpreted as
        # zero r.h.s. by NewtonSolver
        self.newton_solver.Mult(zero, dv_dt)
        add_vector(v, dt, dv_dt, dx_dt)

    def ElasticEnergy(self, x):
        return self.H.GetEnergy(x)

    def KineticEnergy(self, v):
        local_energy = 0.5*self.M.InnerProduct(v, v)
        energy = MPI.COMM_WORLD.allreduce(local_energy, op=MPI.SUM)
        return energy

    def GetElasticEnergyDensity(self, x, w):
        w_coeff = ElasticEnergyCoefficient(self.model, x)
        w.ProjectCoefficient(w_coeff)


def visualize(out, pmesh, deformed_nodes, field,
              field_name='', init_vis=False):
    nodes = deformed_nodes
    owns_nodes = 0

    nodes, owns_nodes = pmesh.SwapNodes(nodes, owns_nodes)

    out.send_text("parallel " + str(num_procs) + " " + str(myid))
    out.send_solution(pmesh, field)

    nodes, owns_nodes = pmesh.SwapNodes(nodes, owns_nodes)

    if (init_vis):
        out.send_text("window_size 400 400")
        out.send_text("window_title '" + field_name)
        if (pmesh.SpaceDimension() == 2):
            out.send_text("view 0 0")
            out.send_text("keys jl")
        out.send_text("keys cm")         # show colorbar and mesh
        # update value-range; keep mesh-extents fixed
        out.send_text("autoscale value")
        out.send_text("pause")
    out.flush()


oper = HyperelasticOperator(fespace, ess_bdr, visc,  mu, K)
if (visualization):
    vis_v = mfem.socketstream("localhost", 19916)
    vis_v.precision(8)
    visualize(vis_v, pmesh, x_gf, v_gf, "Velocity", True)

    MPI.COMM_WORLD.Barrier()
    vis_w = mfem.socketstream("localhost", 19916)
    oper.GetElasticEnergyDensity(x_gf, w_gf)
    vis_w.precision(8)
    visualize(vis_w, pmesh, x_gf, w_gf, "Elastic energy density", True)

ee0 = oper.ElasticEnergy(x_gf)
ke0 = oper.KineticEnergy(v_gf)

if myid == 0:
    print("initial elastic energy (EE) = " + str(ee0))
    print("initial kinetic energy (KE) = " + str(ke0))
    print("initial   total energy (TE) = " + str(ee0 + ke0))

# initialize timers
fom_timer, dmd_training_timer, dmd_prediction_timer = \
        StopWatch(), StopWatch(), StopWatch()

# 8. Perform time-integration (looping over the time iterations, ti, with a
#    time-step dt).
t = 0.
ti = 1
fom_timer.Start()
oper.SetTime(t)
ode_solver.Init(oper)
last_step = False
fom_timer.Stop()

curr_window = 0
dmd_v = []
dmd_x = []
ts = []
dmd_training_timer.Start()
dmd_v.append(DMD(vx.GetBlock(0).Size(),dt))
dmd_x.append(DMD(vx.GetBlock(1).Size(),dt))
dmd_v[curr_window].takeSample(vx.GetBlock(0).GetDataArray(), t)
dmd_x[curr_window].takeSample(vx.GetBlock(1).GetDataArray(), t)
ts.append(t)
dmd_training_timer.Stop()

while not last_step:
    fom_timer.Start()
    dt_real = min(dt, t_final - t)
    t, dt = ode_solver.Step(vx, t, dt_real)
    if (t >= t_final - 1e-8*dt):
        last_step = True
    fom_timer.Stop()

    dmd_training_timer.Start()
    dmd_v[curr_window].takeSample(vx.GetBlock(0).GetDataArray(), t)
    dmd_x[curr_window].takeSample(vx.GetBlock(1).GetDataArray(), t)
    if (last_step or (ti % windowNumSamples) == 0):
        if myid == 0 and rdim != -1 and ef != -1:
            print('Both rdim and ef are set. ef will be ignored')
        if rdim != -1:
            if myid == 0:
                print(f'Creating DMD with rdim: {rdim}')
            dmd_v[curr_window].train(rdim)
            dmd_x[curr_window].train(rdim)
        elif ef != 1:
            if myid == 0:
                print(f'Creating DMD with energy fraction: {ef}')
            dmd_v[curr_window].train(ef)
            dmd_x[curr_window].train(ef)
        if not last_step:
            curr_window += 1
            dmd_v.append(DMD(vx.GetBlock(0).Size(),dt))
            dmd_x.append(DMD(vx.GetBlock(1).Size(),dt))
            dmd_v[curr_window].takeSample(vx.GetBlock(0).GetDataArray(), t)
            dmd_x[curr_window].takeSample(vx.GetBlock(1).GetDataArray(), t)
    ts.append(t)
    dmd_training_timer.Stop()

    if (last_step or (ti % vis_steps) == 0):
        v_gf.Distribute(vx.GetBlock(0))
        x_gf.Distribute(vx.GetBlock(1))

        ee = oper.ElasticEnergy(x_gf)
        ke = oper.KineticEnergy(v_gf)

        text = ("step " + str(ti) + ", t = " + str(t) +
                ", EE = " + "{:g}".format(ee) +
                ", KE = " + "{:g}".format(ke) +
                ", dTE = " + "{:g}".format((ee+ke)-(ee0+ke0)))

        if myid == 0:
            print(text)
        if visualization:
            visualize(vis_v, pmesh, x_gf, v_gf)
            oper.GetElasticEnergyDensity(x_gf, w_gf)
            visualize(vis_w, pmesh, x_gf, w_gf)

    ti = ti + 1

true_solution_v = vx.GetBlock(0)
true_solution_x = vx.GetBlock(1)

# Predict using DMD.
dmd_prediction_timer.Start()
if myid == 0:
    print('Predicting position and velocity using DMD')

curr_window = 0
result_v = dmd_v[curr_window].predict(ts[0]) 
result_x = dmd_x[curr_window].predict(ts[0]) 
initial_dmd_v = mfem.Vector(result_v.getData(), result_v.dim())
initial_dmd_x = mfem.Vector(result_x.getData(), result_x.dim())
v_gf.SetFromTrueDofs(initial_dmd_v)
x_gf.SetFromTrueDofs(initial_dmd_x)

if visit:
    dmd_dc = mfem.VisItDataCollection('DMD_nonlinear_elasticity', pmesh)
    dmd_dc.RegisterField('v',v_gf)
    dmd_dc.RegisterField('x',x_gf)
    dmd_dc.SetCycle(0)
    dmd_dc.SetTime(0.0)
    dmd_dc.Save()

    for i in range(1,len(ts)):
        result_v = dmd_v[curr_window].predict(ts[i]) 
        result_x = dmd_x[curr_window].predict(ts[i]) 
        dmd_solution_v = mfem.Vector(result_v.getData(), result_v.dim())
        dmd_solution_x = mfem.Vector(result_x.getData(), result_x.dim())
        v_gf.SetFromTrueDofs(dmd_solution_v)
        x_gf.SetFromTrueDofs(dmd_solution_x)

        nodes = x_gf
        owns_nodes = 0
        nodes, owns_nodes = pmesh.SwapNodes(nodes, owns_nodes)
        dmd_dc.SetCycle(i)
        dmd_dc.SetTime(ts[i])
        dmd_dc.Save()
        nodes, owns_nodes = pmesh.SwapNodes(nodes, owns_nodes)

        if i % windowNumSamples == 0 and i < len(ts)-1:
            curr_window += 1
else:
    curr_window = len(dmd_v) - 1

result_v = dmd_v[curr_window].predict(t_final) 
result_x = dmd_x[curr_window].predict(t_final) 

dmd_prediction_timer.Stop()

# Calculate the relative error between the DMD final solution and the true solution.
dmd_solution_v = mfem.Vector(result_v.getData(), result_v.dim())
diff_v = mfem.Vector(true_solution_v.Size())
mfem.subtract_vector(dmd_solution_v, true_solution_v, diff_v)
tot_diff_norm_v = np.sqrt(mfem.InnerProduct(diff_v, diff_v))
tot_true_solution_v_norm = np.sqrt(mfem.InnerProduct(true_solution_v, true_solution_v))

dmd_solution_x = mfem.Vector(result_x.getData(), result_x.dim())
diff_x = mfem.Vector(true_solution_x.Size())
mfem.subtract_vector(dmd_solution_x, true_solution_x, diff_x)
tot_diff_norm_x = np.sqrt(mfem.InnerProduct(diff_x, diff_x))
tot_true_solution_x_norm = np.sqrt(mfem.InnerProduct(true_solution_x, true_solution_x))

if myid == 0:
    print("Relative error of DMD velocity (v) at t_final: %f is %.3E" % (t_final, tot_diff_norm_v / tot_true_solution_v_norm))
    print("Relative error of DMD position (x) at t_final: %f is %.3E" % (t_final, tot_diff_norm_x / tot_true_solution_x_norm))
    print("Elapsed time for solving FOM: %e second\n" % fom_timer.duration)
    print("Elapsed time for training DMD: %e second\n" % dmd_training_timer.duration)
    print("Elapsed time for predicting DMD: %e second\n" % dmd_prediction_timer.duration)

#
# if i translate c++ line-by-line, ti seems the second swap does not work...
#

smyid = '{:0>6d}'.format(myid)
mesh_name = "deformed."+smyid
velo_name = "velocity."+smyid
ee_name = "elastic_energy."+smyid

nodes = x_gf
owns_nodes = 0
nodes, owns_nodes = pmesh.SwapNodes(nodes, owns_nodes)
pmesh.Print(mesh_name, 8)
pmesh.SwapNodes(nodes, owns_nodes)

v_gf.Save(velo_name, 8)
oper.GetElasticEnergyDensity(x_gf, w_gf)
w_gf.Save(ee_name,  8)
