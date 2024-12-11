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

//                       libROM MFEM Example: Parametric_Heat_Conduction (adapted from ex16p.cpp)
//
// Compile with: make parametric_heat_conduction
//
// =================================================================================
//
// In these examples, the radius of the interface between different initial temperatures, the
// alpha coefficient, and two center location variables are modified.
//
// For Parametric DMD (ex. 1) (radius & cx & cy, interpolation):
//   rm -rf parameters.txt
//   python3 parametric_heat_conduction.py -r 0.1 -cx 0.1 -cy 0.1 -o 4 -visit -offline -rdim 16
//   python3 parametric_heat_conduction.py -r 0.1 -cx 0.1 -cy 0.5 -o 4 -visit -offline -rdim 16
//   python3 parametric_heat_conduction.py -r 0.1 -cx 0.5 -cy 0.5 -o 4 -visit -offline -rdim 16
//   python3 parametric_heat_conduction.py -r 0.5 -cx 0.1 -cy 0.1 -o 4 -visit -offline -rdim 16
//   python3 parametric_heat_conduction.py -r 0.25 -cx 0.2 -cy 0.4 -o 4 -online -predict
//   python3 parametric_heat_conduction.py -r 0.4 -cx 0.2 -cy 0.3 -o 4 -online -predict
//
// =================================================================================
//
// Description:  This example solves a time dependent nonlinear heat equation
//               problem of the form du/dt = C(u), with a non-linear diffusion
//               operator C(u) = \nabla \cdot (\kappa + \alpha u) \nabla u.
//
//               The example demonstrates the use of nonlinear operators (the
//               class ConductionOperator defining C(u)), as well as their
//               implicit time integration. Note that implementing the method
//               ConductionOperator::ImplicitSolve is the only requirement for
//               high-order implicit (SDIRK) time integration. Optional saving
//               with ADIOS2 (adios2.readthedocs.io) is also illustrated.
'''
import os
import io
import pathlib
import sys
try:
    import mfem.par as mfem
except ModuleNotFoundError:
    msg = "PyMFEM is not installed yet. Install PyMFEM:\n"
    msg += "\tgit clone https://github.com/mfem/PyMFEM.git\n"
    msg += "\tcd PyMFEM\n"
    msg += "\tpython3 setup.py install --with-parallel\n"
    raise ModuleNotFoundError(msg)

from mfem.par import intArray
from os.path import expanduser, join, dirname
import numpy as np
from numpy import sin, cos, exp, sqrt, pi, abs, array, floor, log, sum

sys.path.append("../../build")
import pylibROM.algo as algo
import pylibROM.linalg as linalg
import pylibROM.utils as utils
from pylibROM.python_utils import StopWatch
from pylibROM.mfem import PointwiseSnapshot

class ConductionOperator(mfem.PyTimeDependentOperator):
    def __init__(self, fespace, alpha, kappa, u):
        mfem.PyTimeDependentOperator.__init__(
            self, fespace.GetTrueVSize(), 0.0)
        rel_tol = 1e-8
        self.alpha = alpha
        self.kappa = kappa
        self.T = None
        self.K = None
        self.M = None
        self.fespace = fespace

        self.ess_tdof_list = intArray()
        self.Mmat = mfem.HypreParMatrix()
        self.Kmat = mfem.HypreParMatrix()
        self.M_solver = mfem.CGSolver(fespace.GetComm())
        self.M_prec = mfem.HypreSmoother()
        self.T_solver = mfem.CGSolver(fespace.GetComm())
        self.T_prec = mfem.HypreSmoother()
        self.z = mfem.Vector(self.Height())

        self.M = mfem.ParBilinearForm(fespace)
        self.M.AddDomainIntegrator(mfem.MassIntegrator())
        self.M.Assemble()
        self.M.FormSystemMatrix(self.ess_tdof_list, self.Mmat)

        self.M_solver.iterative_mode = False
        self.M_solver.SetRelTol(rel_tol)
        self.M_solver.SetAbsTol(0.0)
        self.M_solver.SetMaxIter(100)
        self.M_solver.SetPrintLevel(0)
        self.M_prec.SetType(mfem.HypreSmoother.Jacobi)
        self.M_solver.SetPreconditioner(self.M_prec)
        self.M_solver.SetOperator(self.Mmat)

        self.T_solver.iterative_mode = False
        self.T_solver.SetRelTol(rel_tol)
        self.T_solver.SetAbsTol(0.0)
        self.T_solver.SetMaxIter(100)
        self.T_solver.SetPrintLevel(0)
        self.T_solver.SetPreconditioner(self.T_prec)

        self.SetParameters(u)

    def Mult(self, u, u_dt):
        # Compute:
        #  du_dt = M^{-1}*-K(u) for du_dt
        self.Kmat.Mult(u, self.z)
        self.z.Neg()   # z = -z
        self.M_solver.Mult(self.z, du_dt)

    def ImplicitSolve(self, dt, u, du_dt):
        # Solve the equation:
        #    du_dt = M^{-1}*[-K(u + dt*du_dt)]
        #    for du_dt
        if self.T is None:
            self.T = mfem.Add(1.0, self.Mmat, dt, self.Kmat)
            current_dt = dt
            self.T_solver.SetOperator(self.T)
        self.Kmat.Mult(u, self.z)
        self.z.Neg()
        self.T_solver.Mult(self.z, du_dt)

    def SetParameters(self, u):
        u_alpha_gf = mfem.ParGridFunction(self.fespace)
        u_alpha_gf.SetFromTrueDofs(u)
        for i in range(u_alpha_gf.Size()):
            u_alpha_gf[i] = self.kappa + self.alpha * u_alpha_gf[i]

        self.K = mfem.ParBilinearForm(self.fespace)
        u_coeff = mfem.GridFunctionCoefficient(u_alpha_gf)
        self.K.AddDomainIntegrator(mfem.DiffusionIntegrator(u_coeff))
        self.K.Assemble(0)
        self.K.FormSystemMatrix(self.ess_tdof_list, self.Kmat)
        self.T = None

class InitialTemperature(mfem.PyCoefficient):
    def __init__(self, radius_, cx_, cy_):
        self.radius = radius_
        self.cx = cx_
        self.cy = cy_

        mfem.PyCoefficient.__init__(self)
        return
    
    def EvalValue(self, x):
        xx = np.array(x) - np.array([self.cx, self.cy])
        norm2 = np.sqrt(float(np.sum(xx**2)))
        if norm2 < self.radius:
            return 2.0
        return 1.0

if __name__ == "__main__":
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    myid = comm.Get_rank()
    num_procs = comm.Get_size()

    from mfem.common.arg_parser import ArgParser
    parser = ArgParser(description="DMD - MFEM wave equation (ex23) example.")
    parser.add_argument('-m', '--mesh',
                        default='../data/star.mesh',
                        action='store', type=str,
                        help='Mesh file to use.')
    parser.add_argument('-rs', '--refine-serial',
                        action='store', default=2, type=int,
                        help="Number of times to refine the mesh uniformly in serial")
    parser.add_argument('-rp', '--refine-parallel',
                        action='store', default=1, type=int,
                        help="Number of times to refine the mesh uniformly in parallel")
    parser.add_argument('-o', '--order',
                        action='store', default=2, type=int,
                        help="Finite element order (polynomial degree)")
    parser.add_argument('-s', '--ode-solver',
                        action='store', default=3, type=int,
                        help='\n'.join(["ODE solver: 1 - Backward Euler, 2 - SDIRK2, 3 - SDIRK3",
                                    "\t\t 11 - Forward Euler, 12 - RK2, 13 - RK3 SSP, 14 - RK4."]))
    parser.add_argument('-t', '--t-final',
                        action='store', default=0.5, type=float,
                        help="Final time; start time is 0.")
    parser.add_argument("-dt", "--time-step",
                        action='store', default=0.01, type=float,
                        help="Time step.")
    parser.add_argument('-a', '--alpha',
                        action='store', default=0.01, type=float,
                        help='Alpha coefficient')
    parser.add_argument('-k', '--kappa',
                        action='store', default=0.5, type=float,
                        help='Kappa coefficient')
    parser.add_argument("-r", "--radius",
                        action='store', default=0.5, type=float,
                        help="Radius of the interface of initial temperature.")
    parser.add_argument("-cx", "--center_x",
                        action='store', default=0.0, type=float,
                        help="Center offset in the x direction.")
    parser.add_argument("-cy", "--center_y",
                        action='store', default=0.0, type=float,
                        help="Center offset in the y direction.")
    parser.add_argument("-crv", "--crv",
                        action='store', default=0.9, type=float,
                        help="DMD Closest RBF Value.")
    parser.add_argument('-vis', '--visualization',
                        action='store_true', default=True,
                        help='Enable GLVis visualization')
    parser.add_argument('-no-vis', '--no-visualization',
                        action='store_false', dest='visualization',
                        help='Enable GLVis visualization')
    parser.add_argument('-visit', '--visit-datafiles',
                        action='store_true', default=False,
                        help="Save data files for VisIt (visit.llnl.gov) visualization.")
    parser.add_argument("-vs", "--visualization-steps",
                        action='store', default=5,  type=int,
                        help="Visualize every n-th timestep.")
    parser.add_argument("-rdim", "--rdim",
                        action='store', default=-1, type=int,
                        help="Reduced dimension for DMD.")
    parser.add_argument("-offline", "--offline",
                        action='store_true', default=False,
                        help="Enable or disable the offline phase.")
    parser.add_argument("-online", "--online",
                        action='store_true', default=False,
                        help="Enable or disable the online phase.")
    parser.add_argument("-predict", "--predict",
                        action='store_true', default=False,
                        help="Enable or disable DMD prediction.")
    parser.add_argument("-adios2", "--adios2-streams",
                        action='store_true', default=False,
                        help="Save data using adios2 streams.")
    parser.add_argument("-save", "--save",
                        action='store_true', default=False,
                        help="Enable or disable MFEM DOF solution snapshot files).")
    parser.add_argument("-csv", "--csv",
                        action='store_true', default=False, dest='csvFormat',
                        help="Use CSV or HDF format for files output by -save option.")
    parser.add_argument("-hdf", "--hdf",
                        action='store_false', dest='csvFormat',
                        help="Use CSV or HDF format for files output by -save option.")
    parser.add_argument("-out", "--outputfile-name",
                        action='store', default="", type=str,
                        help="Name of the sub-folder to dump files within the run directory.")
    parser.add_argument("-pwsnap", "--pw-snap",
                        action='store_true', default=False,
                        help="Enable or disable writing pointwise snapshots.")
    parser.add_argument("-pwx", "--pwx",
                        action='store', default=0, type=int,
                        help="Number of snapshot points in x")
    parser.add_argument("-pwy", "--pwy",
                        action='store', default=0, type=int,
                        help="Number of snapshot points in y")
    parser.add_argument("-pwz", "--pwz",
                        action='store', default=0, type=int,
                        help="Number of snapshot points in z")
    
    args = parser.parse_args()
    if (myid == 0):
        parser.print_options(args)

    precision           = 8

    save_dofs           = args.save
    basename            = args.outputfile_name
    offline             = args.offline
    online              = args.online
    pointwiseSnapshots  = args.pw_snap
    rdim                = args.rdim
    mesh_file           = expanduser(join(os.path.dirname(__file__),
                                    '..', 'data', args.mesh))
    ode_solver_type     = args.ode_solver
    ser_ref_levels      = args.refine_serial
    par_ref_levels      = args.refine_parallel
    pwx, pwy, pwz       = args.pwx, args.pwy, args.pwz
    order               = args.order
    alpha               = args.alpha
    kappa               = args.kappa
    radius              = args.radius
    cx, cy              = args.center_x, args.center_y
    visit               = args.visit_datafiles
    adios2              = args.adios2_streams
    visualization       = args.visualization
    csvFormat           = args.csvFormat
    dt                  = args.time_step
    t_final             = args.t_final
    vis_steps           = args.visualization_steps
    closest_rbf_val     = args.crv
    predict             = args.predict

    outputPath = "."
    if (save_dofs):
        outputPath = "run"
        if (basename != ""):
            outputPath += "/" + basename

        if (myid == 0):
            pathlib.Path(outputPath).mkdir(parents=True, exist_ok=True)

    check = pointwiseSnapshots ^ offline ^ online ^ save_dofs
    if not check:
        raise RuntimeError("Only one of offline, online, save, or pwsnap must be true!")

    if (offline and (rdim <= 0)):
        raise RuntimeError("rdim must be set.")

    # 3. Read the serial mesh from the given mesh file on all processors. We can
    #    handle triangular, quadrilateral, tetrahedral and hexahedral meshes
    #    with the same code.
    mesh = mfem.Mesh(mesh_file, 1, 1)
    dim = mesh.Dimension()

    # 4. Define the ODE solver used for time integration. Several implicit
    #    singly diagonal implicit Runge-Kutta (SDIRK) methods, as well as
    #    explicit Runge-Kutta methods are available.
    if ode_solver_type == 1:
        ode_solver = mfem.BackwardEulerSolver()
    elif ode_solver_type == 2:
        ode_solver = mfem.SDIRK23Solver(2)
    elif ode_solver_type == 3:
        ode_solver = mfem.SDIRK33Solver()
    elif ode_solver_type == 11:
        ode_solver = mfem.ForwardEulerSolver()
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
    for x in range(par_ref_levels):
        pmesh.UniformRefinement()

    # TODO(kevin): enforce user to install pymfem with gslib?
# #ifndef MFEM_USE_GSLIB
#     if (pointwiseSnapshots) {
#         cout << "To use pointwise snapshots, compile with -mg option" << endl;
#         MFEM_ABORT("Pointwise snapshots aren't available, since the "
#                    "compilation is done without the -mg option");
#     }
    pws = None
    pwsnap = mfem.Vector()
    pwsnap_CAROM = None

    if (pointwiseSnapshots):
        pmesh.EnsureNodes()
        dmdDim = [pwx, pwy, pwz]
        pws = PointwiseSnapshot(dim, dmdDim)
        pws.SetMesh(pmesh)

        snapshotSize = np.prod(dmdDim[:dim])

        pwsnap.SetSize(snapshotSize)
        if (myid == 0):
            pwsnap_CAROM = linalg.Vector(pwsnap.GetDataArray(), True, False)

    # 7. Define the vector finite element space representing the current and the
    #    initial temperature, u_ref.
    fe_coll = mfem.H1_FECollection(order, dim)
    fespace = mfem.ParFiniteElementSpace(pmesh, fe_coll)

    fe_size = fespace.GlobalTrueVSize()
    if (myid == 0):
        print("Number of temperature unknowns: %d" % fe_size)

    u_gf = mfem.ParGridFunction(fespace)

    # 8. Set the initial conditions for u. All boundaries are considered
    #    natural.
    u_0 = InitialTemperature(radius, cx, cy)
    u_gf.ProjectCoefficient(u_0)
    u = mfem.Vector()
    u_gf.GetTrueDofs(u)

    # 9. Initialize the conduction operator and the VisIt visualization.
    oper = ConductionOperator(fespace, alpha, kappa, u)
    u_gf.SetFromTrueDofs(u)

    mesh_name = "%s/parametric_heat_conduction_%f_%f_%f_%f-mesh.%06d" % (outputPath, radius, alpha, cx, cy, myid)
    sol_name = "%s/parametric_heat_conduction_%f_%f_%f_%f-init.%06d" % (outputPath, radius, alpha, cx, cy, myid)

    pmesh.Print(mesh_name, precision)

    output = io.StringIO()
    output.precision = precision
    u_gf.Save(output)
    fid = open(sol_name, 'w')
    fid.write(output.getvalue())
    fid.close()

    visit_name = "%s/Parametric_Heat_Conduction_%f_%f_%f_%f" % (outputPath, radius, alpha, cx, cy)
    visit_dc = mfem.VisItDataCollection(visit_name, pmesh)
    visit_dc.RegisterField("temperature", u_gf)
    if (visit):
        visit_dc.SetCycle(0)
        visit_dc.SetTime(0.0)
        visit_dc.Save()

    # Optionally output a BP (binary pack) file using ADIOS2. This can be
    # visualized with the ParaView VTX reader.
    # TODO(kevin): enforce user to install pymfem with adios2?
#ifdef MFEM_USE_ADIOS2
    if (adios2):
        postfix = mesh_file[len("../data/"):]
        postfix += "_o%d" % order
        postfix += "_solver%d" % ode_solver_type
        collection_name = "%s/parametric_heat_conduction-p-%s.bp" % (outputPath, postfix)

        adios2_dc = mfem.ADIOS2DataCollection(MPI.COMM_WORLD, collection_name, pmesh)
        adios2_dc.SetParameter("SubStreams", "%d" % (num_procs/2) )
        adios2_dc.RegisterField("temperature", u_gf)
        adios2_dc.SetCycle(0)
        adios2_dc.SetTime(0.0)
        adios2_dc.Save()
#endif

    if visualization:
        sol_sock = mfem.socketstream("localhost", 19916)
        if not sol_sock.good():
            print("Unable to connect to GLVis server at localhost:19916")
            visualization = False
            print("GLVis visualization disabled.")
        else:
            sol_sock << "parallel " << num_procs << " " << myid << "\n"
            sol_sock.precision(precision)
            sol_sock << "solution\n" << pmesh << u_gf
            sol_sock << "pause\n"
            sol_sock.flush()
            print(
                "GLVis visualization paused. Press space (in the GLVis window) to resume it.")

#ifdef MFEM_USE_GSLIB
    # TODO(kevin): enforce user to install pymfem with gslib?
    if (pointwiseSnapshots):
        pws.GetSnapshot(u_gf, pwsnap)

        dmd_filename = "snap_%f_%f_%f_%f_0" % (radius, alpha, cx, cy)
        if (myid == 0):
            print("Writing DMD snapshot at step 0, time 0.0")
            pwsnap_CAROM.write(dmd_filename)
#endif

    fom_timer, dmd_training_timer, dmd_prediction_timer = StopWatch(), StopWatch(), StopWatch()

    fom_timer.Start()

    # 10. Perform time-integration (looping over the time iterations, ti, with a
    #     time-step dt).
    ode_solver.Init(oper)
    t = 0.0
    ts = []
    # CAROM::Vector* init = NULL;

    # CAROM::Database *db = NULL;
    if (csvFormat):
        db = utils.CSVDatabase()
    else:
        db = utils.HDFDatabase()

    snap_list = []

    fom_timer.Stop()

    # CAROM::DMD* dmd_u = NULL;

    if (offline):
        dmd_training_timer.Start()

        # 11. Create DMD object and take initial sample.
        u_gf.SetFromTrueDofs(u)
        dmd_u = algo.DMD(u.Size(), dt)
        dmd_u.takeSample(u.GetDataArray(), t)

        if (myid == 0):
            print("Taking snapshot at: %f" % t)

        dmd_training_timer.Stop()

    if (online):
        u_gf.SetFromTrueDofs(u)
        init = linalg.Vector(u.GetDataArray(), u.Size(), True)

    if (save_dofs and (myid == 0)):
        if (csvFormat):
            pathlib.Path("%s/step0" % outputPath).mkdir(parents=True, exist_ok=True)
            db.putDoubleArray(outputPath + "/step0/sol.csv", u.GetDataArray(), u.Size())
        else:
            db.create(outputPath + "/dmd_0.hdf")
            db.putDoubleArray("step0sol", u.GetDataArray(), u.Size())

    ts += [t]
    snap_list += [0]

    last_step = False
    ti = 1
    while (not last_step):
        fom_timer.Start()

        if (t + dt >= t_final - dt / 2.):
            last_step = True

        t, dt = ode_solver.Step(u, t, dt)

        fom_timer.Stop()

        if (offline):
            dmd_training_timer.Start()

            u_gf.SetFromTrueDofs(u)
            dmd_u.takeSample(u.GetDataArray(), t)

            if (myid == 0):
                print("Taking snapshot at: %f" % t)

            dmd_training_timer.Stop()

        if (save_dofs and (myid == 0)):
            if (csvFormat):
                pathlib.Path("%s/step%d" % (outputPath, ti)).mkdir(parents=True, exist_ok=True)
                db.putDoubleArray("%s/step%d/sol.csv" % (outputPath, ti), u.GetDataArray(), u.Size())
            else:
                db.putDoubleArray("step%dsol" % ti, u.GetDataArray(), u.Size())

        ts += [t]
        snap_list += [ti]

        if (last_step or ((ti % vis_steps) == 0)):
            if (myid == 0):
                print("step %d, t = %f" % (ti, t))

            u_gf.SetFromTrueDofs(u)
            if (visualization):
                if sol_sock.good():
                    sol_sock << "parallel " << num_procs << " " << myid << "\n"
                    sol_sock << "solution\n" << pmesh << u_gf
                    # sol_sock << "pause\n"
                    sol_sock.flush()

            if (visit):
                visit_dc.SetCycle(ti)
                visit_dc.SetTime(t)
                visit_dc.Save()

#ifdef MFEM_USE_ADIOS2
            if (adios2):
                adios2_dc.SetCycle(ti)
                adios2_dc.SetTime(t)
                adios2_dc.Save()
#endif

#ifdef MFEM_USE_GSLIB
        if (pointwiseSnapshots):
            pws.GetSnapshot(u_gf, pwsnap)

            dmd_filename = "snap_%f_%f_%f_%f_%d" % (radius, alpha, cx, cy, ti)
            if (myid == 0):
                print("Writing DMD snapshot at step %d, time %f" % (ti, t))
                pwsnap_CAROM.write(dmd_filename)
#endif

        oper.SetParameters(u)

        ti += 1

    if (save_dofs and (myid == 0)):
        if (csvFormat):
            db.putDoubleVector(outputPath + "/tval.csv", ts, len(ts))
            db.putInteger(outputPath + "/numsnap", len(snap_list))
            db.putIntegerArray(outputPath + "/snap_list.csv", snap_list, len(snap_list))
        else:
            db.putDoubleVector("tval", ts, len(ts))
            db.putInteger("numsnap", len(snap_list))
            db.putInteger("snap_bound_size", 0)
            db.putIntegerArray("snap_list", snap_list, len(snap_list))

#ifdef MFEM_USE_ADIOS2
    if (adios2):
        del adios2_dc
#endif

    # 12. Save the final solution in parallel. This output can be viewed later
    #     using GLVis: "glvis -np <np> -m parametric_heat_conduction-mesh -g parametric_heat_conduction-final".
    sol_name = "%s/parametric_heat_conduction_%f_%f_%f_%f-final.%06d" % (outputPath, radius, alpha, cx, cy, myid)
    output = io.StringIO()
    output.precision = precision
    u_gf.Save(output)
    fid = open(sol_name, 'w')
    fid.write(output.getvalue())
    fid.close()

    # 13. Calculate the DMD modes.
    if (offline or online):
        if (offline):
            if (myid == 0):
                print("Creating DMD with rdim: %d" % rdim)

            dmd_training_timer.Start()

            dmd_u.train(rdim)

            dmd_training_timer.Stop()

            dmd_u.save("%s/%f_%f_%f_%f" % (outputPath, radius, alpha, cx, cy))

            if (myid == 0):
                with open("parameters.txt", "ab") as f:
                    np.savetxt(f, [radius, alpha, cx, cy])

        if (online):
            if (myid == 0):
                print("Creating DMD using the rdim of the offline phase")

            param_hist = np.loadtxt("parameters.txt")
            param_hist = param_hist.reshape((int(param_hist.size / 4), 4))

            dmd_paths = []
            param_vectors = []

            for curr_param in param_hist:
                curr_radius, curr_alpha, curr_cx, curr_cy = curr_param

                dmd_paths += ["%s/%f_%f_%f_%f" % (outputPath, curr_radius, curr_alpha, curr_cx, curr_cy)]
                param_vectors += [linalg.Vector([curr_radius, curr_alpha, curr_cx, curr_cy], False)]

            desired_param = linalg.Vector([radius, alpha, cx, cy], False)

            dmd_training_timer.Start()

            dmd_u = algo.getParametricDMD(algo.DMD, param_vectors, dmd_paths, desired_param,
                                            "G", "LS", closest_rbf_val)

            dmd_u.projectInitialCondition(init)

            dmd_training_timer.Stop()
            del desired_param

        if (predict):
            true_solution_u = mfem.Vector(u.GetDataArray(), u.Size())

            dmd_prediction_timer.Start()

            # 14. Predict the state at t_final using DMD.
            if (myid == 0):
                print("Predicting temperature using DMD at: %f" % ts[0])

            result_u = dmd_u.predict(ts[0])
            initial_dmd_solution_u = mfem.Vector(result_u.getData(), result_u.dim())
            u_gf.SetFromTrueDofs(initial_dmd_solution_u)

            visit_name = "%s/DMD_Parametric_Heat_Conduction_%f_%f_%f_%f" % (outputPath, radius, alpha, cx, cy)
            dmd_visit_dc = mfem.VisItDataCollection(visit_name, pmesh)
            dmd_visit_dc.RegisterField("temperature", u_gf)
            if (visit):
                dmd_visit_dc.SetCycle(0)
                dmd_visit_dc.SetTime(0.0)
                dmd_visit_dc.Save()

            del result_u

            if (visit):
                for i, tsi in enumerate(ts):
                    if ((i == len(ts) - 1) or ((i % vis_steps) == 0)):
                        result_u = dmd_u.predict(tsi)
                        if (myid == 0):
                            print("Predicting temperature using DMD at: %f" % tsi)

                        dmd_solution_u = mfem.Vector(result_u.getData(), result_u.dim())
                        u_gf.SetFromTrueDofs(dmd_solution_u)

                        dmd_visit_dc.SetCycle(i)
                        dmd_visit_dc.SetTime(tsi)
                        dmd_visit_dc.Save()

                        del result_u

            dmd_prediction_timer.Stop()

            result_u = dmd_u.predict(t_final)

            # 15. Calculate the relative error between the DMD final solution and the true solution.
            dmd_solution_u = mfem.Vector(result_u.getData(), result_u.dim())
            diff_u = mfem.Vector(true_solution_u.Size())
            mfem.subtract_vector(dmd_solution_u, true_solution_u, diff_u)

            tot_diff_norm_u = sqrt(mfem.InnerProduct(MPI.COMM_WORLD, diff_u, diff_u))
            tot_true_solution_u_norm = sqrt(mfem.InnerProduct(MPI.COMM_WORLD,
                                                true_solution_u, true_solution_u))

            if (myid == 0):
                print("Relative error of DMD temperature (u) at t_final: %f is %f" % (t_final,
                    tot_diff_norm_u / tot_true_solution_u_norm))

                print("Elapsed time for predicting DMD: %e second" % dmd_prediction_timer.duration)

            del result_u

        if (myid == 0):
            print("Elapsed time for training DMD: %e second" % dmd_training_timer.duration)

    if (myid == 0):
        print("Elapsed time for solving FOM: %e second" % fom_timer.duration)

    # 16. Free the used memory.
    del ode_solver
    del pmesh
    if (offline):
        del dmd_u

    if not csvFormat:
        db.close()

#ifdef MFEM_USE_GSLIB
    del pws
    del pwsnap_CAROM
#endif

    MPI.Finalize()
