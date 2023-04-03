ARCH = None
from petsc4py.lib import ImportPETSc  # noqa: E402
PETSc = ImportPETSc(ARCH)
PETSc._initialize()
del PETSc
del ImportPETSc
del ARCH
