ARCH = None
from petsc4py.lib import ImportPETSc
PETSc = ImportPETSc(ARCH)
del ImportPETSc, ARCH
PETSc._initialize()
del PETSc
