ARCH = None
from petsc4py.lib import ImportPETSc
PETSc = ImportPETSc(ARCH)
del  ARCH, ImportPETSc
PETSc._initialize()
del PETSc
