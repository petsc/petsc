PYTHON           = python
PYTHON_INCLUDE   = ${shell ${PYTHON} -c 'from distutils import sysconfig; print(sysconfig.get_python_inc())'}
PYTHON_LIBDIR    = ${shell ${PYTHON} -c 'from distutils import sysconfig; print(sysconfig.get_config_var("LIBDIR"))'}
PYTHON_LIB       = ${shell ${PYTHON} -c 'import sys; print("python"+sys.version[:3])'}
PETSC4PY_INCLUDE = ${shell ${PYTHON} -c 'import petsc4py; print(petsc4py.get_include())'}


ALL: lib

CFLAGS   = -I${PYTHON_INCLUDE} -I${PETSC4PY_INCLUDE}
SOURCEC  = python.c
OBJSC    = python.o
LIBBASE  = libpetscsnes
MANSEC   = SNES
LOCDIR   = src/snes/impls/python/

include ${PETSC_DIR}/conf/base
include ${PETSC_DIR}/conf/test

