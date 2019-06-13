
'''
Ex2 from PETSc example files implemented for PETSc4py.
https://www.mcs.anl.gov/petsc/petsc-current/src/ksp/ksp/examples/tutorials/ex2.c.html
By: Miguel Arriaga


Solves a linear system in parallel with KSP.
Input parameters include:
    -view_exact_sol   : write exact solution vector to stdout
    -m <mesh_x>       : number of mesh points in x-direction
    -n <mesh_n>       : number of mesh points in y-direction


Concepts: KSP^basic parallel example
Concepts: KSP^Laplacian, 2d
Concepts: Laplacian, 2d
Processors: n

Vec            x,b,u;    # approx solution, RHS, exact solution 
Mat            A;        # linear system matrix 
KSP            ksp;      # linear solver context 
PetscReal      norm;     # norm of solution error
'''
import sys
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc

import numpy as np

comm = PETSc.COMM_WORLD
size = comm.getSize()
rank = comm.getRank()

OptDB = PETSc.Options()
m  = OptDB.getInt('m', 8)
n  = OptDB.getInt('n', 7)

''' - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Compute the matrix and right-hand-side vector that define
        the linear system, Ax = b.
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - '''
'''
    Create parallel matrix, specifying only its global dimensions.
    When using MatCreate(), the matrix format can be specified at
    runtime. Also, the parallel partitioning of the matrix is
    determined by PETSc at runtime.

    Performance tuning note:  For problems of substantial size,
    preallocation of matrix memory is crucial for attaining good
    performance. See the matrix chapter of the users manual for details.
'''

A = PETSc.Mat().create(comm=comm)
A.setSizes((m*n,m*n))
A.setFromOptions()
A.setPreallocationNNZ((5,5)) 

'''
    Currently, all PETSc parallel matrix formats are partitioned by
    contiguous chunks of rows across the processors.  Determine which
    rows of the matrix are locally owned.
'''
Istart,Iend = A.getOwnershipRange()

'''
    Set matrix elements for the 2-D, five-point stencil in parallel.
    - Each processor needs to insert only elements that it owns
    locally (but any non-local elements will be sent to the
    appropriate processor during matrix assembly).
    - Always specify global rows and columns of matrix entries.

    Note: this uses the less common natural ordering that orders first
    all the unknowns for x = h then for x = 2h etc; Hence you see J = Ii +- n
    instead of J = I +- m as you might expect. The more standard ordering
    would first do all variables for y = h, then y = 2h etc.
'''

for Ii in range(Istart,Iend):
    v = -1.0
    i = int(Ii/n)
    j = int(Ii - i*n)

    if (i>0):
        J = Ii - n
        A.setValues(Ii,J,v,addv=True)
    if (i<m-1):
        J = Ii + n
        A.setValues(Ii,J,v,addv=True)
    if (j>0):
        J = Ii - 1
        A.setValues(Ii,J,v,addv=True)
    if (j<n-1):
        J = Ii + 1
        A.setValues(Ii,J,v,addv=True)

    v = 4.0
    A.setValues(Ii,Ii,v,addv=True)

'''
    Assemble matrix, using the 2-step process:
    MatAssemblyBegin(), MatAssemblyEnd()
    Computations can be done while messages are in transition
    by placing code between these two statements.
'''

A.assemblyBegin(A.AssemblyType.FINAL)
A.assemblyEnd(A.AssemblyType.FINAL)
''' A is symmetric. Set symmetric flag to enable ICC/Cholesky preconditioner '''

A.setOption(A.Option.SYMMETRIC,True)

'''
    Create parallel vectors.
    - We form 1 vector from scratch and then duplicate as needed.
    - When using VecCreate(), VecSetSizes and VecSetFromOptions()
    in this example, we specify only the
    vector's global dimension; the parallel partitioning is determined
    at runtime.
    - When solving a linear system, the vectors and matrices MUST
    be partitioned accordingly.  PETSc automatically generates
    appropriately partitioned matrices and vectors when MatCreate()
    and VecCreate() are used with the same communicator.
    - The user can alternatively specify the local vector and matrix
    dimensions when more sophisticated partitioning is needed
    (replacing the PETSC_DECIDE argument in the VecSetSizes() statement
    below).
'''

u = PETSc.Vec().create(comm=comm)
u.setSizes(m*n)
u.setFromOptions()

b = u.duplicate()
x = b.duplicate()

'''
    Set exact solution; then compute right-hand-side vector.
    By default we use an exact solution of a vector with all
    elements of 1.0;  
'''
u.set(1.0)
b = A(u)

'''
    View the exact solution vector if desired
'''
flg = OptDB.getBool('view_exact_sol', False)
if flg:
    u.view()

''' - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            Create the linear solver and set various options
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - '''
ksp = PETSc.KSP().create(comm=comm)

'''
    Set operators. Here the matrix that defines the linear system
    also serves as the preconditioning matrix.
'''
ksp.setOperators(A,A)

'''
    Set linear solver defaults for this problem (optional).
    - By extracting the KSP and PC contexts from the KSP context,
    we can then directly call any KSP and PC routines to set
    various options.
    - The following two statements are optional; all of these
    parameters could alternatively be specified at runtime via
    KSPSetFromOptions().  All of these defaults can be
    overridden at runtime, as indicated below.
'''
rtol=1.e-2/((m+1)*(n+1))
ksp.setTolerances(rtol=rtol,atol=1.e-50)

'''
Set runtime options, e.g.,
    -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
These options will override those specified above as long as
KSPSetFromOptions() is called _after_ any other customization
routines.
'''
ksp.setFromOptions()

''' - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Solve the linear system
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - '''

ksp.solve(b,x)

''' - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Check the solution and clean up
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - '''
x = x - u # x.axpy(-1.0,u)
norm = x.norm(PETSc.NormType.NORM_2)
its = ksp.getIterationNumber()

'''
    Print convergence information.  PetscPrintf() produces a single
    print statement from all processes that share a communicator.
    An alternative is PetscFPrintf(), which prints to a file.
'''
if norm > rtol*10:
    PETSc.Sys.Print("Norm of error {}, Iterations {}".format(norm,its),comm=comm)
else:
    if size==1:
        PETSc.Sys.Print("- Serial OK",comm=comm)
    else:
        PETSc.Sys.Print("- Parallel OK",comm=comm)
