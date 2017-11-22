ARCH = None
from petsc4py.lib import ImportPETSc
PETSc = ImportPETSc(ARCH)
PETSc._initialize()
del PETSc
del ImportPETSc
del ARCH
