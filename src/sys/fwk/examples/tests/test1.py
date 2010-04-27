import sys, petsc4py
petsc4py.init(sys.argv)

from petsc4py import PETSc

fwk = PETSc.Fwk().create()
fwk.registerComponent("./testcomponents.py:TestIIIA")
fwk.viewConfigurationOrder()
fwk.configure(1)

del fwk
