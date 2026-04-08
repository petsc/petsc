"""
This example creates a hierarchical matrix arising from the kernel function

   K(x, y) = 1 / (0.01 + ||x - y||_2)

where x and y are points in R^3. The kernel is defined as a Python callable
and passed together with point coordinates to `Mat.createHtoolFromKernel`.

The example demonstrates:
  * defining a kernel functor in Python;
  * passing coordinate spaces for target and source points;
  * converting the hierarchical matrix to a dense matrix for verification.
"""

import sys
import numpy
import petsc4py
petsc4py.init(sys.argv)
from mpi4py import MPI
from petsc4py import PETSc

N = PETSc.Options().getInt("-N", 1000) # number of points (global)
dim = 3 # spatial dimension

# Generate N points in R^3 distributed uniformly along a line
local_rows, _ = PETSc.Sys.splitOwnership(N, comm=PETSc.COMM_WORLD)
coords = numpy.linspace(
    (1.0, 2.0, 3.0),
    (10.0, 20.0, 30.0),
    N,
    dtype=PETSc.RealType,
)
# Compute global offset for this rank
offset = MPI.COMM_WORLD.exscan(local_rows, op=MPI.SUM)
if PETSc.COMM_WORLD.rank == 0:
    offset = 0
# Local coordinates for this rank
local_coords = coords[offset : offset + local_rows]

# The kernel is a Python callable with signature:
#   kernel(sdim, M, N, rows, cols, v, ctx)
#
# Parameters
# ----------
# sdim : int
#     Spatial dimension of the coordinates
# M, N : int
#     Number of target/source points in this submatrix block
# rows : ndarray of int, shape (M,)
#     Global indices of the target points
# cols : ndarray of int, shape (N,)
#     Global indices of the source points
# v : ndarray, shape (M, N)
#     Output array to be filled with kernel values (column-major order)
# ctx : object
#     User context; here it is the full global coordinate array

def kernel(sdim, M, N, rows, cols, v, ctx):
    """Evaluate 1 / (0.01 + ||x - y||) for all (target, source) pairs."""
    gcoords = ctx # global coordinate array, shape (N_global, sdim)
    for i in range(M):
        x = gcoords[rows[i]]
        for j in range(N):
            y = gcoords[cols[j]]
            diff = x - y
            v[i, j] = 1.0 / (0.01 + numpy.sqrt(numpy.dot(diff, diff)))

# Create the MatHtool matrix
A = PETSc.Mat()
A.createHtoolFromKernel(
    [
        [local_rows, N],
        [local_rows, N],
    ],            # size: ((local_rows, global_rows), (local_cols, global_cols))
    dim,
    local_coords, # local target coordinates
    local_coords, # local source coordinates (same for square matrix)
    kernel,
    coords,       # kernelctx: full global coordinates for index lookup
    comm=PETSc.COMM_WORLD
)
A.setFromOptions()
epsilon = A.getHtoolEpsilon()
A.assemble()
A.viewFromOptions('-A_view')

iss = A.getHtoolPermutationSource()
ist = A.getHtoolPermutationTarget()

# Assemble the equivalent MatDense matrix for comparison
D = PETSc.Mat().createDense(size=[[local_rows, N], [local_rows, N]], comm=PETSc.COMM_WORLD)
rows = numpy.arange(offset, offset + local_rows, dtype=PETSc.IntType)
cols = numpy.arange(N, dtype=PETSc.IntType)
# Allocate full local block
v = numpy.zeros((local_rows, N), order='C', dtype=PETSc.RealType)
# Single kernel call for all local rows
kernel(dim, local_rows, N, rows, cols, v, coords)
# Insert everything at once
D.setValues(rows, cols, v)
D.assemble()

x, y_htool = A.createVecs()
x.setRandom()
y_dense_htool = D.createVecLeft()
y_dense = D.createVecLeft()

A.mult(x, y_htool)
A.convert('dense').mult(x, y_dense_htool)
D.mult(x, y_dense)

y_dense_htool.axpy(-1.0, y_htool)
rel_err = y_dense_htool.norm() / y_htool.norm()

y_htool.axpy(-1.0, y_dense)
comp_err = y_htool.norm() / y_dense.norm()

PETSc.Sys.Print(f"Relative errors ||y_dense_Htool - y_Htool|| / ||y_Htool|| = {rel_err:.2e}")
PETSc.Sys.Print(f"                    ||y_Htool - y_dense||  /  ||y_dense|| = {comp_err:.2e}")
if rel_err > 1.0e-10:
    raise ValueError(f"Relative error too large: {rel_err}")
if comp_err > epsilon:
    raise ValueError(f"Compression error too large: {comp_err}")
