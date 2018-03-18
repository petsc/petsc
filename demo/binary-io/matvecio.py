try: range = xrange
except NameError: pass

import sys, petsc4py
petsc4py.init(sys.argv)

from petsc4py import PETSc

m, n  = 16, 32

A = PETSc.Mat().create(PETSc.COMM_WORLD)
A.setSizes([m*n, m*n])
A.setFromOptions()
A.setUp()
Istart, Iend = A.getOwnershipRange()
for I in range(Istart, Iend):
    A[I,I] = 4
    i = I//n
    if i>0  : J = I-n; A[I,J] = -1
    if i<m-1: J = I+n; A[I,J] = -1
    j = I-i*n
    if j>0  : J = I-1; A[I,J] = -1
    if j<n-1: J = I+1; A[I,J] = -1
A.assemblyBegin()
A.assemblyEnd()

x, y = A.createVecs()
x.set(1)
A.mult(x,y)

# save
viewer = PETSc.Viewer().createBinary('matrix-A.dat', 'w')
viewer(A)
viewer = PETSc.Viewer().createBinary('vector-x.dat', 'w')
viewer(x)
viewer = PETSc.Viewer().createBinary('vector-y.dat', 'w')
viewer(y)

# load
viewer = PETSc.Viewer().createBinary('matrix-A.dat', 'r')
B = PETSc.Mat().load(viewer)
viewer = PETSc.Viewer().createBinary('vector-x.dat', 'r')
u = PETSc.Vec().load(viewer)
viewer = PETSc.Viewer().createBinary('vector-y.dat', 'r')
v = PETSc.Vec().load(viewer)

# check
assert B.equal(A)
assert x.equal(u)
assert y.equal(v)
