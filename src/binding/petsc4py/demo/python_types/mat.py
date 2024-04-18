# ------------------------------------------------------------------------
#
#  Poisson problem. This problem is modeled by the partial
#  differential equation
#
#          -Laplacian(u) = 1,  0 < x,y < 1,
#
#  with boundary conditions
#
#           u = 0  for  x = 0, x = 1, y = 0, y = 1
#
#  A finite difference approximation with the usual 5-point stencil
#  is used to discretize the boundary value problem to obtain a
#  nonlinear system of equations. The problem is solved in a 2D
#  rectangular domain, using distributed arrays (DAs) to partition
#  the parallel grid.
#
# ------------------------------------------------------------------------

# We first import petsc4py and sys to initialize PETSc
import sys
import petsc4py

petsc4py.init(sys.argv)

# Import the PETSc module
from petsc4py import PETSc


# Here we define a class representing the discretized operator
# This allows us to apply the operator "matrix-free"
class Poisson2D:
    def __init__(self, da):
        self.da = da
        self.localX = da.createLocalVec()

    # This is the method that PETSc will look for when applying
    # the operator. `X` is the PETSc input vector, `Y` the output vector,
    # while `mat` is the PETSc matrix holding the PETSc datastructures.
    def mult(self, mat, X, Y):
        # Grid sizes
        mx, my = self.da.getSizes()
        hx, hy = (1.0 / m for m in [mx, my])

        # Bounds for the local part of the grid this process owns
        (xs, xe), (ys, ye) = self.da.getRanges()

        # Map global vector to local vectors
        self.da.globalToLocal(X, self.localX)

        # We can access the vector data as NumPy arrays
        x = self.da.getVecArray(self.localX)
        y = self.da.getVecArray(Y)

        # Loop on the local grid and compute the local action of the operator
        for j in range(ys, ye):
            for i in range(xs, xe):
                u = x[i, j]  # center
                u_e = u_w = u_n = u_s = 0
                if i > 0:
                    u_w = x[i - 1, j]  # west
                if i < mx - 1:
                    u_e = x[i + 1, j]  # east
                if j > 0:
                    u_s = x[i, j - 1]  # south
                if j < ny - 1:
                    u_n = x[i, j + 1]  # north
                u_xx = (-u_e + 2 * u - u_w) * hy / hx
                u_yy = (-u_n + 2 * u - u_s) * hx / hy
                y[i, j] = u_xx + u_yy

    # This is the method that PETSc will look for when the diagonal of the matrix is needed.
    def getDiagonal(self, mat, D):
        mx, my = self.da.getSizes()
        hx, hy = (1.0 / m for m in [mx, my])
        (xs, xe), (ys, ye) = self.da.getRanges()

        d = self.da.getVecArray(D)

        # Loop on the local grid and compute the diagonal
        for j in range(ys, ye):
            for i in range(xs, xe):
                d[i, j] = 2 * hy / hx + 2 * hx / hy

    # The class can contain other methods that PETSc won't use
    def formRHS(self, B):
        b = self.da.getVecArray(B)
        mx, my = self.da.getSizes()
        hx, hy = (1.0 / m for m in [mx, my])
        (xs, xe), (ys, ye) = self.da.getRanges()
        for j in range(ys, ye):
            for i in range(xs, xe):
                b[i, j] = 1 * hx * hy


# Access the option database and read options from the command line
OptDB = PETSc.Options()
nx, ny = OptDB.getIntArray(
    'grid', (16, 16)
)  # Read `-grid <int,int>`, defaults to 16,16

# Create the distributed memory implementation for structured grid
da = PETSc.DMDA().create([nx, ny], stencil_width=1)

# Create vectors to hold the solution and the right-hand side
x = da.createGlobalVec()
b = da.createGlobalVec()

# Instantiate an object of our Poisson2D class
pde = Poisson2D(da)

# Create a PETSc matrix of type Python using `pde` as context
A = PETSc.Mat().create(comm=da.comm)
A.setSizes([x.getSizes(), b.getSizes()])
A.setType(PETSc.Mat.Type.PYTHON)
A.setPythonContext(pde)
A.setUp()

# Create a Conjugate Gradient Krylov solver
ksp = PETSc.KSP().create()
ksp.setType(PETSc.KSP.Type.CG)

# Use diagonal preconditioning
ksp.getPC().setType(PETSc.PC.Type.JACOBI)

# Allow command-line customization
ksp.setFromOptions()

# Assemble right-hand side and solve the linear system
pde.formRHS(b)
ksp.setOperators(A)
ksp.solve(b, x)

# Here we programmatically visualize the solution
if OptDB.getBool('plot', True):
    # Modify the option database: keep the X window open for 1 second
    OptDB['draw_pause'] = 1

    # Obtain a viewer of type DRAW
    draw = PETSc.Viewer.DRAW(x.comm)

    # View the vector in the X window
    draw(x)

# We can also visualize the solution by command line options
# For example, we can dump a VTK file with:
#
#     $ python poisson2d.py -plot 0 -view_solution vtk:sol.vts:
#
# or obtain the same visualization as programmatically done above as:
#
#     $ python poisson2d.py -plot 0 -view_solution draw -draw_pause 1
#
x.viewFromOptions('-view_solution')
