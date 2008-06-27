PYTHON           = python
PYTHON_INCLUDE   = ${shell ${PYTHON} -c 'from distutils import sysconfig; print sysconfig.get_python_inc()'}
PYTHON_LIBDIR    = ${shell ${PYTHON} -c 'from distutils import sysconfig; print sysconfig.get_config_var("LIBDIR")'}
PYTHON_LIB       = ${shell ${PYTHON} -c 'import sys; print "python"+sys.version[:3]'}
PETSC4PY_INCLUDE = ${shell ${PYTHON} -c 'import petsc4py; print petsc4py.get_include()'}


ALL: lib

CFLAGS   = -I${PYTHON_INCLUDE} -I${PETSC4PY_INCLUDE}
FFLAGS   =
DIRS     = 
LOCDIR   = src/ksp/ksp/impls/python/
SOURCEH  = python.h
SOURCEC  = python.c
OBJSC    = python.o
LIBBASE  = libpetscksp
MANSEC   = KSP

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test
