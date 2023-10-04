
'''
Ex23 from PETSc example files implemented for PETSc4py.
https://petsc.org/release/src/ksp/ksp/tutorials/ex23.c.html
By: Miguel Arriaga

Solves a tridiagonal linear system.

Vec            x, b, u;           approx solution, RHS, exact solution
Mat            A;                 linear system matrix
KSP            ksp;               linear solver context
PC             pc;                preconditioner context
PetscReal      norm;              norm of solution error

'''
import sys
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc

import numpy as np

comm = PETSc.COMM_WORLD
size = comm.getSize()
rank = comm.getRank()
n = 12 # Size of problem
tol = 1E-11 # Tolerance of Result. tol=1000.*PETSC_MACHINE_EPSILON;

'''
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Compute the matrix and right-hand-side vector that define
    the linear system, Ax = b.
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

Create vectors.  Note that we form 1 vector from scratch and
then duplicate as needed. For this simple case let PETSc decide how
many elements of the vector are stored on each processor. The second
argument to VecSetSizes() below causes PETSc to decide.
'''

x = PETSc.Vec().create(comm=comm)
x.setSizes(n)
x.setFromOptions()

b = x.duplicate()
u = x.duplicate()

'''
Identify the starting and ending mesh points on each
processor for the interior part of the mesh. We let PETSc decide
above.
'''

rstart,rend = x.getOwnershipRange()
nlocal = x.getLocalSize()

'''
Create matrix.  When using MatCreate(), the matrix format can
be specified at runtime.

Performance tuning note:  For problems of substantial size,
preallocation of matrix memory is crucial for attaining good
performance. See the matrix chapter of the users manual for details.

We pass in nlocal as the "local" size of the matrix to force it
to have the same parallel layout as the vector created above.
'''

A = PETSc.Mat().create(comm=comm)
A.setSizes(n,nlocal)
A.setFromOptions()
A.setUp()

'''
Assemble matrix.

The linear system is distributed across the processors by
chunks of contiguous rows, which correspond to contiguous
sections of the mesh on which the problem is discretized.
For matrix assembly, each processor contributes entries for
the part that it owns locally.
'''

col = np.zeros(3,dtype=PETSc.IntType)
value = np.zeros(3,dtype=PETSc.ScalarType)

if not rstart:
    rstart = 1
    i = 0; col[0] = 0; col[1] = 1; value[0] = 2.0; value[1] = -1.0
    A.setValues(i,col[0:2],value[0:2])

if rend == n:
    rend = n-1
    i = n-1; col[0] = n-2; col[1] = n-1; value[0] = -1.0; value[1] = 2.0
    A.setValues(i,col[0:2],value[0:2])


''' Set entries corresponding to the mesh interior '''
value[0] = -1.0; value[1] = 2.0; value[2] = -1.0
for i in range(rstart,rend):
    col[0] = i-1; col[1] = i; col[2] = i+1
    A.setValues(i,col,value)


''' Assemble the matrix '''
A.assemblyBegin(A.AssemblyType.FINAL)
A.assemblyEnd(A.AssemblyType.FINAL)

''' Set exact solution; then compute right-hand-side vector. '''
u.set(1.0)
b = A(u)

'''
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Create the linear solver and set various options
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - '''
'''
Create linear solver context
'''
ksp = PETSc.KSP().create()

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
    - The following four statements are optional; all of these
      parameters could alternatively be specified at runtime via
      KSPSetFromOptions();
'''
pc = ksp.getPC()
pc.setType('jacobi')
ksp.setTolerances(rtol=1.e-7)

'''
Set runtime options, e.g.,
-ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
These options will override those specified above as long as
KSPSetFromOptions() is called _after_ any other customization
routines.
'''
ksp.setFromOptions()

'''
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Solve the linear system
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - '''

''' Solve linear system '''
ksp.solve(b,x)

'''
View solver info; we could instead use the option -ksp_view to
print this info to the screen at the conclusion of KSPSolve().
'''
# ksp.view()

'''
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Check solution and clean up
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - '''

''' Check the error '''
x = x - u # x.axpy(-1.0,u)
norm = x.norm(PETSc.NormType.NORM_2)
its = ksp.getIterationNumber()
if norm > tol:
    PETSc.Sys.Print("Norm of error {}, Iterations {}\n".format(norm,its),comm=comm)
else:
    if size==1:
        PETSc.Sys.Print("- Serial OK",comm=comm)
    else:
        PETSc.Sys.Print("- Parallel OK",comm=comm)
