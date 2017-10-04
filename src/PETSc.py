import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ndarray size changed")
ARCH = None
from petsc4py.lib import ImportPETSc
PETSc = ImportPETSc(ARCH)
PETSc._initialize()
del PETSc
del ImportPETSc
del ARCH
