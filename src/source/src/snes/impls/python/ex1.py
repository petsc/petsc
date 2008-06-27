#!/usr/bin/env python

import sys, petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc

opts = PETSc.Options()
opts['snes_python'] = 'mysolver,MyNewton'

N = 10
J = PETSc.Mat().create(PETSc.COMM_SELF)
J.setSizes(N);
J.setFromOptions()
J.setUp()

x, f = J.getVecs()

def function(snes, x, F):
    f = x * x
    f.copy(F)

def jacobian(snes, x, J, P):
    P.zeroEntries()
    diag = 2 * x
    P.setDiagonal(diag)
    P.assemble()
    if J != P: J.assemble()

snes = PETSc.SNES()
snes.create(PETSc.COMM_SELF)
snes.setFunction(function, f)
snes.setJacobian(jacobian, J, J)

snes.setFromOptions()
x.setRandom()
snes.solve(None, x)

del opts['snes_python']
