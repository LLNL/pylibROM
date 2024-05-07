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
#//               pylibROM MFEM Example: parametric ROM for linear elasticity problem (adapted from ex2p.py)
#//
#//
#// Description:  This example code demonstrates the use of pyMFEM and pylibROM to
#//               define a simple projection-based reduced order model of a
#//				 simple linear elasticity problem describing a multi-material
#//				 cantilever beam.
#//
#//               The example highlights three distinct ROM processes, i.e.,
#//               offline, merge, and online. The offline phase runs the full
#//               order model and stores the snapshot data in an HDF file. You
#//               can run as many offline phases as you wish to sample the
#//               parameter space. The merge phase reads all the snapshot files,
#//               builds a global reduced basis, and stores the basis in an HDF
#//               file. The online phase reads the basis, builds the ROM
#//               operator, solves the reduced order system, and lifts the
#//               solution to the full order space.
#//
#// Offline phase: python linear_elasticity_global_rom.py -offline -id 0 -nu 0.2
#//                python linear_elasticity_global_rom.py -offline -id 1 -nu 0.4
#//
#// Merge phase:   python linear_elasticity_global_rom.py -merge -ns 2
#//
#// NB: to evaluate relative error, call this before the online phase:
#//                python linear_elasticity_global_rom.py -offline -id 2 -nu 0.3
#//
#// Online phase:  python linear_elasticity_global_rom.py -online -id 3 -nu 0.3
#//
#// This example runs in parallel with MPI, by using the same number of MPI ranks
#// in all phases (offline, merge, online).


'''
   MFEM example 2

   See c++ version in the MFEM library for more detail 

   How to run:
      mpirun -np 2 python2.7 <arguments>
  
   Example of arguments:
      ex2p.py -m star.mesh
      ex2p.py -m square-disc.mesh
      ex2p.py -m escher.mesh
      ex2p.py -m fichera.mesh
      ex2p.py -m beam-tri.mesh -o 2 -sys   
      ex2p.py -m beam-quad.mesh -o 3 -elast
      ex2p.py -m beam-quad.mesh -o 3 -sc
'''
import numpy as np
from os.path import expanduser, join, dirname
import sys, io
from mfem import path
from mfem.common.arg_parser import ArgParser
import mfem.par as mfem
from mpi4py import MPI
comm = MPI.COMM_WORLD
num_procs = comm.size
myid = comm.rank

# libROM libraries
from pylibROM.python_utils.StopWatch import StopWatch
import pylibROM.linalg as la
from pylibROM.mfem import ComputeCtAB


parser = ArgParser(description='linear elastisity global rom')
parser.add_argument('-m', '--mesh',
                    default='beam-tri.mesh',
                    action='store', type=str,
                    help='Mesh file to use.')
parser.add_argument('-o', '--order',
                    action='store', default=1, type=int,
                    help="Finite element order (polynomial degree) or -1 for isoparametric space.")
parser.add_argument('-id','--id',
                    action='store',default=0,type=int,
                    help='Parametric id')
parser.add_argument('-ns','--nset',
                    action='store',default=0,type=int,
                    help='Number of parametric snapshot sets')
parser.add_argument('-sc', '--static-condensation',
                    action='store_false',default=False,
                    help="Enable static condensation.")
parser.add_argument('-f', '--ext-force',
                    action='store',default=-1.0e-2,type=float,
                    help="External force applied at end")
parser.add_argument('-E', '--youngs_modulus',
                    action='store',default=2.5,type=float,
                    help="Young's modulus")
parser.add_argument('-nu', '--poisson_ratio',
                    action='store',default=.25,type=float,
                    help="Poisson's ratio")
parser.add_argument('-elast', '--amg_elast',
                    action='store_true',
                    help='Use the special AMG elasticity solver (GM/LN approaches)',
                    dest='amg_elast', default=False)
parser.add_argument('-sys', '--amg-for-systems',
                    action='store_false',
                    help='Use  standard AMG for systems (unknown approach).',
                    dest='amg_elast', default=True)
parser.add_argument('-nodes', '--by_nodes',
                    action='store_true',
                    dest='reorder_space',default=False,
                    help="Use byNODES ordering of vector space instead of byVDIM")
parser.add_argument('-d','--device',
                    action='store',default='cpu',
                    help='Device configuration string')
parser.add_argument('-visit', '--visit-datafiles',
                    action='store_true',dest='visit',default=True,
                    help='Save data files for VisIt visualization')
parser.add_argument('-vis', '--visualization',
                    action='store_true',default=True,
                    help='Enable GLVis visualization')
parser.add_argument('-fom', '--fom',
                    action='store_true',default=False,
                    help='Enable or diable the fom phase')
parser.add_argument('-offline', '--offline',
                    action='store_true',default=False,
                    help='Enable or diable the offline phase.')
parser.add_argument('-online', '--online',
                    action='store_true',default=False,
                    help='Enable or diable the online phase.')
parser.add_argument('-merge', '--merge',
                    action='store_true',default=False,
                    help='Enable or diable the merge phase.')

args = parser.parse_args()
mesh_file           = args.mesh
order               = args.order
id                  = args.id
nsets               = args.nset
static_cond         = args.static_condensation
ext_force           = args.ext_force
E                   = args.youngs_modulus
nu                  = args.poisson_ratio
amg_elast           = args.amg_elast
reorder_space       = args.reorder_space
device_config       = args.device
visit               = args.visit
visualization       = args.visualization
fom                 = args.fom
offline             = args.offline
online              = args.online
merge               = args.merge
precision           = 8

if (myid == 0):
    parser.print_options(args)

device = mfem.Device(device_config)
if myid == 0:
    device.Print()

#  3. Read the (serial) mesh from the given mesh file on all processors.  We
#     can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
#     and volume meshes with the same code.
meshfile = expanduser(join(dirname(__file__), '..', 'data', mesh_file))
mesh = mfem.Mesh(meshfile, 1, 1)
dim = mesh.Dimension()
if (mesh.attributes.Max() < 2 or mesh.bdr_attributes.Max() < 2):
    if (myid == 0):
        print('\n'.join(['Input mesh should have at least two materials and',
                         'two boundary attributes! (See schematic in ex2.cpp)']))
    sys.exit()

#  4. Select the order of the finite element discretization space. For NURBS
#     meshes, we increase the order by degree elevation.
if (mesh.NURBSext and order > mesh.NURBSext.GetOrder()):
    mesh.DegreeElevate(order - mesh.NURBSext.GetOrder())

#  5. Refine the serial mesh on all processors to increase the resolution. In
#     this example we do 'ref_levels' of uniform refinement. We choose
#     'ref_levels' to be the largest number that gives a final mesh with no
#     more than 1,000 elements.
ref_levels = int(np.floor(np.log(1000./mesh.GetNE())/np.log(2.)/dim))
for x in range(ref_levels):
    mesh.UniformRefinement()

#  6. Define a parallel mesh by a partitioning of the serial mesh. Refine
#     this mesh further in parallel to increase the resolution. Once the
#     parallel mesh is defined, the serial mesh can be deleted.
pmesh = mfem.ParMesh(MPI.COMM_WORLD, mesh)
del mesh
par_ref_levels = 1
for l in range(par_ref_levels):
    pmesh.UniformRefinement()

#   7. Define a parallel finite element space on the parallel mesh. Here we
#      use vector finite elements, i.e. dim copies of a scalar finite element
#      space. We use the ordering by vector dimension (the last argument of
#      the FiniteElementSpace constructor) which is expected in the systems
#      version of BoomerAMG preconditioner. For NURBS meshes, we use the
#      (degree elevated) NURBS space associated with the mesh nodes.
use_nodal_fespace = pmesh.NURBSext and not amg_elast
if use_nodal_fespace:
    fespace = pmesh.GetNodes().FESpace()
else:
    fec = mfem.H1_FECollection(order, dim)
    fespace = mfem.ParFiniteElementSpace(pmesh, fec, dim,
                                         mfem.Ordering.byVDIM)

size = fespace.GlobalTrueVSize()
if (myid == 0):
    print("Number of finite element unknowns: " + str(size))

# 8. Determine the list of true (i.e. parallel conforming) essential
#    boundary dofs. In this example, the boundary conditions are defined by
#    marking only boundary attribute 1 from the mesh as essential and
#    converting it to a list of true dofs.
ess_bdr = mfem.intArray(pmesh.bdr_attributes.Max())
ess_tdof_list = mfem.intArray()
ess_bdr.Assign(0)
ess_bdr[0] = 1
fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list)

# 9. Initiate ROM related variables
max_num_snapshots = 100
update_right_SV = False
isIncremental = False
basisName = 'basis'
basisFileName = basisName + f'{id}'
solveTimer, assembleTimer, mergeTimer = StopWatch(), StopWatch(), StopWatch()

# 10. Set BasisGenerate if offline
if offline:
    options = la.Options(fespace.GetTrueVSize(), max_num_snapshots, update_right_SV)
    generator = la.BasisGenerator(options, isIncremental, basisFileName)

# 11. The merge phase
if merge:
    mergeTimer.Start()
    options = la.Options(fespace.GetTrueVSize(), max_num_snapshots, update_right_SV)
    generator = la.BasisGenerator(options,isIncremental, basisName)
    for paramID in range(nsets):
        snapshot_filename = f'{basisName}{paramID}_snapshot'
        generator.loadSamples(snapshot_filename,'snapshot')
    generator.endSamples()
    mergeTimer.Stop()
    if myid==0:
        print(f'Elapsed time for merging and building ROM basis: {mergeTimer.duration} second')

    sys.exit()

#  12.Set up the parallel linear form b(.) which corresponds to the
#     right-hand side of the FEM linear system. In this case, b_i equals the
#     boundary integral of f*phi_i where f represents a "pull down" force on
#     the Neumann part of the boundary and phi_i are the basis functions in
#     the finite element fespace. The force is defined by the object f, which
#     is a vector of Coefficient objects. The fact that f is non-zero on
#     boundary attribute 2 is indicated by the use of piece-wise constants
#     coefficient for its last component.
f = mfem.VectorArrayCoefficient(dim)
for i in range(dim-1):
    f.Set(i, mfem.ConstantCoefficient(0.0))

pull_force = mfem.Vector([0]*pmesh.bdr_attributes.Max())
pull_force[1] = ext_force
f.Set(dim-1, mfem.PWConstCoefficient(pull_force))


b = mfem.ParLinearForm(fespace)
b.AddBoundaryIntegrator(mfem.VectorBoundaryLFIntegrator(f))
if myid==0:
    print('r.h.s. ...')
b.Assemble()

# 13. Define the solution vector x as a parallel finite element grid
#     function corresponding to fespace. Initialize x with initial guess of
#     zero, which satisfies the boundary conditions.
x = mfem.GridFunction(fespace)
x.Assign(0.0)

# 14. Set up the parallel bilinear form a(.,.) on the finite element space
#     corresponding to the linear elasticity integrator with piece-wise
#     constants coefficient lambda and mu.

assembleTimer.Start()
lamb = mfem.Vector(pmesh.attributes.Max())
lamb.Assign((E*nu)/((1+nu)*(1-2*nu)))
lamb[0] = lamb[1] * 50
lambda_func = mfem.PWConstCoefficient(lamb)

mu = mfem.Vector(pmesh.attributes.Max())
mu.Assign(E/(2*(1+nu)))
mu[0] = mu[1]*50
mu_func = mfem.PWConstCoefficient(mu)

a = mfem.ParBilinearForm(fespace)
a.AddDomainIntegrator(mfem.ElasticityIntegrator(lambda_func, mu_func))

#  15. Assemble the parallel bilinear form and the corresponding linear
#      system, applying any necessary transformations such as: parallel
#      assembly, eliminating boundary conditions, applying conforming
#      constraints for non-conforming AMR, static condensation, etc.
if (myid == 0):
    print('matrix...')
if (static_cond):
    a.EnableStaticCondensation()
a.Assemble()

A = mfem.HypreParMatrix()
B = mfem.Vector()
X = mfem.Vector()
a.FormLinearSystem(ess_tdof_list, x, b, A, X, B)
if (myid == 0):
    print('...done')
    print("Size of linear system: " + str(A.GetGlobalNumRows()))
assembleTimer.Stop()

# 16. The offline phase
if fom or offline:

    #  17. Define and apply a parallel PCG solver for A X = B with the BoomerAMG
    #      preconditioner from hypre.
    amg = mfem.HypreBoomerAMG(A)
    if (amg_elast and not a.StaticCondensationIsEnabled()):
        amg.SetElasticityOptions(fespace)
    else:
        amg.SetSystemsOptions(dim, reorder_space)
    pcg = mfem.HyprePCG(A)
    pcg.SetTol(1e-8)
    pcg.SetMaxIter(500)
    pcg.SetPrintLevel(2)
    pcg.SetPreconditioner(amg)
    solveTimer.Start()
    pcg.Mult(B, X)
    solveTimer.Stop()
    
    # 18. Take and write snapshot for ROM
    if offline:
        addSample = generator.takeSample(X.GetDataArray())
        generator.writeSnapshot()

# 19. The online phase
if online:
    # 20. read the reduced basis
    assembleTimer.Start()
    reader = la.BasisReader(basisName)
    spatialbasis = reader.getSpatialBasis()
    numRowRB = spatialbasis.numRows()
    numColumnRB = spatialbasis.numColumns()
    print(f'On rank {myid}, spatial basis dimension is {numRowRB} x {numColumnRB}')

    # 21. form inverse ROM  operator
    invReducedA = la.Matrix(numColumnRB, numColumnRB, False)
    ComputeCtAB(A, spatialbasis, spatialbasis, invReducedA)
    invReducedA.invert()

    B_carom = la.Vector(B.GetDataArray(),  True, False)
    X_carom = la.Vector(X.GetDataArray(),  True, False)
    reducedRHS = spatialbasis.transposeMult(B_carom)
    reducedSol = la.Vector(numColumnRB, False)

    assembleTimer.Stop()

    # 22. solve ROM
    solveTimer.Start()
    invReducedA.mult(reducedRHS, reducedSol)
    solveTimer.Stop()

    # 23. reconstruct FOM state
    spatialbasis.mult(reducedSol, X_carom)


#  24. Recover the parallel grid function corresponding to X. This is the
#      local finite element solution on each processor.
a.RecoverFEMSolution(X, b, x)

#  25. For non-NURBS meshes, make the mesh curved based on the finite element
#      space. This means that we define the mesh elements through a fespace
#      based transformation of the reference element.  This allows us to save
#      the displaced mesh as a curved mesh when using high-order finite
#      element displacement field. We assume that the initial mesh (read from
#      the file) is not higher order curved mesh compared to the chosen FE
#      space.
if (not use_nodal_fespace):
    pmesh.SetNodalFESpace(fespace)

#  26. Save in parallel the displaced mesh and the inverted solution (which
#      gives the backward displacements to the original grid). This output
#      can be viewed later using GLVis: "glvis -np <np> -m mesh -g sol".

#smyid = '{:0>6d}'.format(myid)
#mesh_name = "mesh."+smyid
#sol_name = "sol."+smyid

if fom or offline:
    mesh_name = f'mesh_f{ext_force}_fom.{myid:06d}'
    sol_name = f'sol_f{ext_force}_fom.{myid:06d}'
if online:
    mesh_name = f'mesh_f{ext_force}_rom.{myid:06d}'
    sol_name = f'sol_f{ext_force}_rom.{myid:06d}'
    sol_name_fom = f'sol_f{ext_force}_fom.{myid:06d}'

nodes = pmesh.GetNodes()
nodes += x
x *= -1


# check if this is correct
pmesh.Print(mesh_name, precision)
np.savetxt(sol_name,x.GetDataArray(),fmt='%.16f')


# 27. Calculate the relative error of the ROM prediction compared to FOM
if online:
    x_fom = mfem.Vector(x.Size())
    x_fom.Load(sol_name_fom,x_fom.Size())
    diff_x = mfem.Vector(x.Size())
    mfem.subtract_vector(x,x_fom,diff_x)

    # get norms
    tot_diff_norm_x = np.sqrt(mfem.InnerProduct(comm,diff_x,diff_x))
    tot_x_fom_norm = np.sqrt(mfem.InnerProduct(comm,x_fom,x_fom))

    if myid==0:
        print(f'Relative error of ROM for E={E} and nu={nu} is {tot_diff_norm_x/tot_x_fom_norm}')

# 28. Save data in the VisIt format
if visit:
    if offline:
        dc = mfem.VisItDataCollection('Example_linear_elastic',pmesh)
        dc.SetPrecision(precision)
        dc.RegisterField('solution',x)
        dc.Save()
    elif online:
        dc = mfem.VisItDataCollection('Example_linear_elastic_rom',pmesh)
        dc.SetPrecision(precision)
        dc.RegisterField('solution',x)
        dc.Save()



# 29. Send the above data by socket to a GLVis server.  Use the "n" and "b"
#      keys in GLVis to visualize the displacements.

if (visualization):
    sol_sock = mfem.socketstream("localhost", 19916)
    sol_sock.send_text("parallel " + str(num_procs) + " " + str(myid))
    sol_sock.precision(8)
    sol_sock.send_solution(pmesh,  x)

# 30. Print timing info
if myid==0:
    if fom or offline:
        print(f'Elapsed time for assembling FOM: {assembleTimer.duration} second')
        print(f'Elapsed time for solving FOM: {solveTimer.duration} second')
    if online:
        print(f'Elapsed time for assembling ROM: {assembleTimer.duration} second')
        print(f'Elapsed time for solving ROM: {solveTimer.duration} second')

