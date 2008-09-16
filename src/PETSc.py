from petsc4py.lib import ImportPETSc
PETSc = ImportPETSc()
del ImportPETSc
PETSc._initialize()
del PETSc
