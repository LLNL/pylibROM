'''
******************************************************************************
*
* Copyright (c) 2013-2023, Lawrence Livermore National Security, LLC
* and other libROM project developers. See the top-level COPYRIGHT
* file for details.
*
* SPDX-License-Identifier: (Apache-2.0 OR MIT)
*
*****************************************************************************

This code is adapted from PyMFEM/examples/ex18p.py, and is a mirror of 
its cpp version located at libROM/examples/dmd/dg_euler.cpp

Below is the description from libROM/examples/dmd/dg_euler.cpp:

// Compile with: make dg_euler

// =================================================================================
//
// Sample runs and results for adaptive DMD:
//
// Command 1:
//   mpirun -np 8 python dg_euler.py -p 1 -rs 1 -rp 1 -o 5 -s 6 -tf 0.1 -visit
//
// Output 1:
//   Relative error of DMD density (dens) at t_final: 0.1 is 0.00015272589
//   Relative error of DMD x-momentum (x_mom) at t_final: 0.1 is 2.8719908e-05
//   Relative error of DMD y-momentum (y_mom) at t_final: 0.1 is 8.9435003e-05
//   Relative error of DMD energy (e) at t_final: 0.1 is 6.85403e-05
//
// Command 2:
//   mpirun -np 8 python dg_euler.py -p 2 -rs 2 -rp 1 -o 1 -s 3 -tf 0.1 -visit
//
// Output 2:
//   Relative error of DMD density (dens) at t_final: 0.1 is 1.573349e-06
//   Relative error of DMD x-momentum (x_mom) at t_final: 0.1 is 4.3846865e-05
//   Relative error of DMD y-momentum (y_mom) at t_final: 0.1 is 0.0026493438
//   Relative error of DMD energy (e) at t_final: 0.1 is 1.7326842e-06
//
// Command 3:
//   mpirun -np 8 python dg_euler.py -p 2 -rs 2 -rp 1 -o 1 -s 3 -visit
//
// Output 3:
//   Relative error of DMD density (dens) at t_final: 2 is 0.00022777614
//   Relative error of DMD x-momentum (x_mom) at t_final: 2 is 0.00022107792
//   Relative error of DMD y-momentum (y_mom) at t_final: 2 is 0.00030374609
//   Relative error of DMD energy (e) at t_final: 2 is 0.0002277899
//
// =================================================================================
//
// Sample runs and results for nonuniform DMD:
//
// Command 1:
//   mpirun -np 8 python dg_euler.py -p 1 -rs 1 -rp 1 -o 5 -s 6 -tf 0.1 -nonunif -visit
//
// Output 1:
//   Relative error of DMD density (dens) at t_final: 0.1 is 0.00015499558
//   Relative error of DMD x-momentum (x_mom) at t_final: 0.1 is 4.5300074e-05
//   Relative error of DMD y-momentum (y_mom) at t_final: 0.1 is 0.0034796374
//   Relative error of DMD energy (e) at t_final: 0.1 is 7.0110651e-05
//
// Command 2:
//   mpirun -np 8 python dg_euler.py -p 2 -rs 2 -rp 1 -o 1 -s 3 -tf 0.1 -nonunif -visit
//
// Output 2:
//   Relative error of DMD density (dens) at t_final: 0.1 is 4.1676355e-07
//   Relative error of DMD x-momentum (x_mom) at t_final: 0.1 is 4.4263729e-05
//   Relative error of DMD y-momentum (y_mom) at t_final: 0.1 is 0.0017438412
//   Relative error of DMD energy (e) at t_final: 0.1 is 8.3869658e-07
//
// Command 3:
//   mpirun -np 8 python dg_euler.py -p 2 -rs 2 -rp 1 -o 1 -s 3 -nonunif -visit
//
// Output 3:
//   Relative error of DMD density (dens) at t_final: 0.1 is 7.9616991e-07
//   Relative error of DMD x-momentum (x_mom) at t_final: 0.1 is 0.00011741735
//   Relative error of DMD y-momentum (y_mom) at t_final: 0.1 is 0.016937741
//   Relative error of DMD energy (e) at t_final: 0.1 is 2.6258626e-06
//
// =================================================================================
//
// Description:  This example code solves the compressible Euler system of
//               equations, a model nonlinear hyperbolic PDE, with a
//               discontinuous Galerkin (DG) formulation.
//
//               Specifically, it solves for an exact solution of the equations
//               whereby a vortex is transported by a uniform flow. Since all
//               boundaries are periodic here, the method's accuracy can be
//               assessed by measuring the difference between the solution and
//               the initial condition at a later time when the vortex returns
//               to its initial location.
//
//               Note that as the order of the spatial discretization increases,
//               the timestep must become smaller. This example currently uses a
//               simple estimate derived by Cockburn and Shu for the 1D RKDG
//               method. An additional factor can be tuned by passing the --cfl
//               (or -c shorter) flag.
//
//               The example demonstrates user-defined bilinear and nonlinear
//               form integrators for systems of equations that are defined with
//               block vectors, and how these are used with an operator for
//               explicit time integrators. In this case the system also
//               involves an external approximate Riemann solver for the DG
//               interface flux. It also demonstrates how to use GLVis for
//               in-situ visualization of vector grid functions.
//
//               We recommend viewing examples 9, 14 and 17 before viewing this
//               example.

'''
import mfem.par as mfem

from dg_euler_common import FE_Evolution, InitialCondition, \
                            RiemannSolver, DomainIntegrator, FaceIntegrator
from mfem.common.arg_parser import ArgParser

from os.path import expanduser, join, dirname
import numpy as np
from numpy import sqrt, pi, cos, sin, hypot, arctan2
from scipy.special import erfc
from ctypes import *

# Equation constant parameters.(using globals to share them with dg_euler_common)
import dg_euler_common
from pylibROM.python_utils.StopWatch import StopWatch
from pylibROM.algo import NonuniformDMD, AdaptiveDMD

# 1. Initialize MPI.from mpi4py import MPI

from mpi4py import MPI
num_procs = MPI.COMM_WORLD.size
myid = MPI.COMM_WORLD.rank

parser = ArgParser(description='dg_euler')
parser.add_argument('-m', '--mesh',
                    default='periodic-square.mesh',
                    action='store', type=str,
                    help='Mesh file to use.')
parser.add_argument('-p', '--problem',
                    action='store', default=1, type=int,
                    help='Problem setup to use. See options in velocity_function().')
parser.add_argument('-rs', '--refine_serial',
                    action='store', default=0, type=int,
                    help="Number of times to refine the mesh uniformly before parallel.")
parser.add_argument('-rp', '--refine_parallel',
                    action='store', default=1, type=int,
                    help="Number of times to refine the mesh uniformly after parallel.")
parser.add_argument('-o', '--order',
                    action='store', default=3, type=int,
                    help="Finite element order (polynomial degree)")
parser.add_argument('-s', '--ode_solver',
                    action='store', default=4, type=int,
                    help="ODE solver: 1 - Forward Euler,\n\t" +
                    "            2 - RK2 SSP, 3 - RK3 SSP, 4 - RK4, 6 - RK6.")
parser.add_argument('-tf', '--t_final',
                    action='store', default=2.0, type=float,
                    help="Final time; start time is 0.")
parser.add_argument("-dt", "--time_step",
                    action='store', default=-0.01, type=float,
                    help="Time step.")
parser.add_argument('-c', '--cfl_number',
                    action='store', default=0.3, type=float,
                    help="CFL number for timestep calculation.")
parser.add_argument('-vis', '--visualization',
                    action='store_true',
                    help='Enable GLVis visualization')
parser.add_argument('-visit', '--visit-datafiles',
                    action='store_true', default=False,
                    help='Save data files for VisIt (visit.llnl.gov) visualize.')
parser.add_argument('-vs', '--visualization_steps',
                    action='store', default=50, type=float,
                    help="Visualize every n-th timestep.")

# additional args for DMD
parser.add_argument('-ef','--energy_fraction',
                    action='store',default=0.9999, type=float,
                    help='Energy fraction for DMD')
parser.add_argument('-rdim','--rdim',
                    action='store',default=-1,type=int,
                    help='Reduced dimension for DMD')
parser.add_argument('-crbf','--crbf',
                    action='store',default=0.9,type=float,
                    help='Closest RBF value')
parser.add_argument('-nonunif','--nonunif',dest='nonunif',
                    action='store_true',help='Use NonuniformDMD')


# assign args
args                = parser.parse_args()
mesh                = args.mesh
ser_ref_levels      = args.refine_serial
par_ref_levels      = args.refine_parallel
order               = args.order
ode_solver_type     = args.ode_solver
t_final             = args.t_final
dt                  = args.time_step
cfl                 = args.cfl_number
visualization       = args.visualization
visit               = args.visit_datafiles
vis_steps           = args.visualization_steps
ef                  = args.energy_fraction
rdim                = args.rdim
crbf                = args.crbf
nonunif             = args.nonunif

if myid == 0:
    parser.print_options(args)

device = mfem.Device('cpu')
if myid == 0:
    device.Print()

dg_euler_common.num_equation = 4
dg_euler_common.specific_heat_ratio = 1.4
dg_euler_common.gas_constant = 1.0
dg_euler_common.problem = args.problem
num_equation = dg_euler_common.num_equation


# 3. Read the mesh from the given mesh file. This example requires a 2D
#    periodic mesh, such as ../data/periodic-square.mesh.
meshfile = expanduser(join(dirname(__file__), '..', 'data', mesh))
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
for lev in range(par_ref_levels):
    pmesh.UniformRefinement()

# 7. Define the discontinuous DG finite element space of the given
#    polynomial order on the refined mesh.
fec = mfem.DG_FECollection(order, dim)
# Finite element space for a scalar (thermodynamic quantity)
fes = mfem.ParFiniteElementSpace(pmesh, fec)
# Finite element space for a mesh-dim vector quantity (momentum)
dfes = mfem.ParFiniteElementSpace(pmesh, fec, dim, mfem.Ordering.byNODES)
# Finite element space for all variables together (total thermodynamic state)
vfes = mfem.ParFiniteElementSpace(
    pmesh, fec, num_equation, mfem.Ordering.byNODES)

assert fes.GetOrdering() == mfem.Ordering.byNODES, "Ordering must be byNODES"
glob_size = vfes.GlobalTrueVSize()
if myid == 0:
    print("Number of unknowns: " + str(glob_size))

# 8. Define the initial conditions, save the corresponding mesh and grid
#    functions to a file. This can be opened with GLVis with the -gc option.
#    The solution u has components {density, x-momentum, y-momentum, energy}.
#    These are stored contiguously in the BlockVector u_block.

offsets = [k*vfes.GetNDofs() for k in range(num_equation+1)]
offsets = mfem.intArray(offsets)
u_block = mfem.BlockVector(offsets)

#  Momentum grid function on dfes for visualization.
mom = mfem.ParGridFunction(dfes, u_block,  offsets[1])

#  Initialize the state.
u0 = InitialCondition(num_equation)
sol = mfem.ParGridFunction(vfes, u_block.GetData())
sol.ProjectCoefficient(u0)

smyid = '{:0>6d}'.format(myid)
pmesh.Print("vortex-mesh."+smyid, 8)
for k in range(num_equation):
    uk = mfem.ParGridFunction(fes, u_block.GetBlock(k).GetData())
    sol_name = "vortex-" + str(k) + "-init."+smyid
    uk.Save(sol_name, 8)

# 9. Set up the nonlinear form corresponding to the DG discretization of the
#    flux divergence, and assemble the corresponding mass matrix.
Aflux = mfem.MixedBilinearForm(dfes, fes)
Aflux.AddDomainIntegrator(DomainIntegrator(dim))
Aflux.Assemble()

A = mfem.ParNonlinearForm(vfes)
rsolver = RiemannSolver()
ii = FaceIntegrator(rsolver, dim)
A.AddInteriorFaceIntegrator(ii)

# 10. Define the time-dependent evolution operator describing the ODE
#     right-hand side, and perform time-integration (looping over the time
#     iterations, ti, with a time-step dt).
euler = FE_Evolution(vfes, A, Aflux.SpMat())

if (visualization):
    MPI.COMM_WORLD.Barrier()
    sout = mfem.socketstream("localhost", 19916)
    sout.send_text("parallel " + str(num_procs) + " " + str(myid))
    sout.precision(8)
    sout.send_solution(pmesh, mom)
    sout.send_text("pause")
    sout.flush()
    if myid == 0:
        print("GLVis visualization paused.")
        print(" Press space (in the GLVis window) to resume it.")
visit_dc = mfem.VisItDataCollection('DG_Euler',pmesh)
visit_dc.RegisterField('solution',mom)
if visit:
    visit_dc.SetCycle(0)
    visit_dc.SetTime(0.0)
    visit_dc.Save()

# Determine the minimum element size.
my_hmin = 0
if (cfl > 0):
    my_hmin = min([pmesh.GetElementSize(i, 1) for i in range(pmesh.GetNE())])
hmin = MPI.COMM_WORLD.allreduce(my_hmin, op=MPI.MIN)

# initialize timers
fom_timer, dmd_training_timer, dmd_prediction_timer = \
        StopWatch(), StopWatch(), StopWatch()

fom_timer.Start()
t = 0.0
ts = []
euler.SetTime(t)
ode_solver.Init(euler)
fom_timer.Stop()

if (cfl > 0):
    #  Find a safe dt, using a temporary vector. Calling Mult() computes the
    #  maximum char speed at all quadrature points on all faces.
    z = mfem.Vector(A.Width())
    A.Mult(sol, z)
    max_char_speed = MPI.COMM_WORLD.allreduce(dg_euler_common.max_char_speed, op=MPI.MAX) 
    dg_euler_common.max_char_speed = max_char_speed 
    dt = cfl * hmin / dg_euler_common.max_char_speed / (2*order+1)

#- DMD setup
# Initialize dmd_vars = [dmd_dens, dmd_x_mom, dmd_y_mom, dmd_e]
dmd_training_timer.Start()
if nonunif:
    dmd_vars = [NonuniformDMD(u_block.GetBlock(i).Size()) 
                for i in range(4)]
else:
    dmd_vars = [AdaptiveDMD(u_block.GetBlock(i).Size(),dt,'G','LS',crbf) \
                for i in range(4)]
for i in range(4):
    dmd_vars[i].takeSample(u_block.GetBlock(i).GetDataArray(),t)
ts += [t]
dmd_training_timer.Stop()

# Integrate in time.
done = False
ti = 0
while not done:

    fom_timer.Start()

    dt_real = min(dt, t_final - t)
    t, dt_real = ode_solver.Step(sol, t, dt_real)

    if (cfl > 0):
        max_char_speed = MPI.COMM_WORLD.allreduce(
            dg_euler_common.max_char_speed, op=MPI.MAX)
        dg_euler_common.max_char_speed = max_char_speed
        dt = cfl * hmin / dg_euler_common.max_char_speed / (2*order+1)

    ti = ti+1
    done = (t >= t_final - 1e-8*dt)

    fom_timer.Stop()
    
    #- DMD take sample
    dmd_training_timer.Start()
    for i in range(4):
        dmd_vars[i].takeSample(u_block.GetBlock(i).GetDataArray(),t)
    ts.append(t)
    dmd_training_timer.Stop()


    if (done or ti % vis_steps == 0):
        if myid == 0:
            print("time step: " + str(ti) + ", time: " + "{:g}".format(t))
        if (visualization):
            sout.send_text("parallel " + str(num_procs) + " " + str(myid))
            sout.send_solution(pmesh, mom)
            sout.flush()
        if visit:
            visit_dc.SetCycle(ti)
            visit_dc.SetTime(t)
            visit_dc.Save()


if myid == 0:
    print("done")

# 11. Save the final solution. This output can be viewed later using GLVis:
#     "glvis -np 4 -m vortex-mesh -g vortex-1-final".

for k in range(num_equation):
    uk = mfem.ParGridFunction(fes, u_block.GetBlock(k).GetData())
    sol_name = "vortex-" + str(k) + "-final."+smyid
    uk.Save(sol_name, 8)

# 12. Compute the L2 solution error summed for all components.
if (t_final == 2.0):
    error = sol.ComputeLpError(2., u0)
    if myid == 0:
        print("Solution error: " + "{:g}".format(error))

# 13. Calculate the DMD modes
if myid==0 and rdim != -1 and ef != -1:
    print('Both rdim and ef are set. ef will be ignored')

dmd_training_timer.Start()

if rdim != -1:
    if myid==0:
        print(f'Creating DMD with rdim: {rdim}')
    for dmd_var in dmd_vars:
        dmd_var.train(rdim)
elif ef != -1:
    if myid == 0:
        print(f'Creating DMD with energy fraction: {ef}')
    for dmd_var in dmd_vars:
        dmd_var.train(ef)

dmd_training_timer.Stop()

true_solution_vars = [u_block.GetBlock(i) for i in range(4)]

# 14. Predict the state at t_final using DMD
dmd_prediction_timer.Start()
if myid == 0:
    print('Predicting density, momentum, and energy using DMD')

result_vars = [dmd_var.predict(ts[0]) for dmd_var in dmd_vars]
initial_dmd_sols = [mfem.Vector(result_var.getData(), result_var.dim()) for result_var in result_vars]
for i in range(4):
    block = u_block.GetBlock(i)
    block = initial_dmd_sols[i]
    #u_block.Update(initial_dmd_sols[i],offsets[i])

dmd_visit_dc = mfem.VisItDataCollection('DMD_DG_Euler', pmesh)
dmd_visit_dc.RegisterField('solution',mom)
if (visit):
    dmd_visit_dc.SetCycle(0)
    dmd_visit_dc.SetTime(0.0)
    dmd_visit_dc.Save()

if visit:
    for i in range(len(ts)):
        if (i==(len(ts)-2)) or (i%vis_steps==0):
            result_vars = [var.predict(ts[i]) for var in dmd_vars]
            dmd_sols = [mfem.Vector(result_var.getData(),result_var.dim()) for result_var in result_vars]
            for k in range(4):
                block = u_block.GetBlock(i)
                block = dmd_sols[k]

            dmd_visit_dc.SetCycle(i)
            dmd_visit_dc.SetTime(ts[i])
            dmd_visit_dc.Save()
dmd_prediction_timer.Stop()
result_vars = [dmd_var.predict(t_final) for dmd_var in dmd_vars]

# 15. Calculate the relative error between the DMD final solution and the true solution.
dmd_solution_vars = [mfem.Vector(result_var.getData(),result_var.dim()) for result_var in result_vars]
diff_vars = [mfem.Vector(result_var.dim()) for result_var in result_vars]
for i in range(4):
    mfem.subtract_vector(dmd_solution_vars[i],true_solution_vars[i],\
                         diff_vars[i])
tot_diff_norm_vars = [sqrt(mfem.InnerProduct(MPI.COMM_WORLD,diff_var,diff_var)) for diff_var in diff_vars]
tot_true_solution_norm_vars = [sqrt(mfem.InnerProduct(MPI.COMM_WORLD,true_solution_var,true_solution_var))\
                                       for true_solution_var in true_solution_vars]

if myid==0:
    var_names = ['dens', 'x_mom', 'y_mom', 'e']
    for i in range(len(var_names)):
        rel_error = tot_diff_norm_vars[i]/tot_true_solution_norm_vars[i]
        print(f'Relative error of DMD {var_names[i]} at t_final: {t_final} is {rel_error:.10f}')

    print(f'Elapsed time for solving FOM: {fom_timer.duration:.6e}')
    print(f'Elapsed time for training DMD: {dmd_training_timer.duration:.6e}')
    print(f'Elapsed time for predicting DMD: {dmd_prediction_timer.duration:.6e}')

