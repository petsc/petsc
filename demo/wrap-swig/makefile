.PHONY: source build test clean

ALL: build

#

PYTHON = python
PYTHON_INCLUDE   = ${shell ${PYTHON} -c 'from distutils import sysconfig; print(sysconfig.get_python_inc())'}
PYTHON_LIBDIR    = ${shell ${PYTHON} -c 'from distutils import sysconfig; print(sysconfig.get_config_var("LIBDIR"))'}
PYTHON_LIB       = ${shell ${PYTHON} -c 'from distutils import sysconfig; print("python"+sysconfig.get_python_version())'}
PETSC4PY_INCLUDE = ${shell ${PYTHON} -c 'import petsc4py; print(petsc4py.get_include())'}

#

CFLAGS  = -I${PYTHON_INCLUDE} -I${PETSC4PY_INCLUDE}
SOURCEC = _Bratu3D.c Bratu3DApp.c
OBJSC   = _Bratu3D.o Bratu3DApp.o

include ${PETSC_DIR}/conf/base

#


SWIG = swig
source: _Bratu3D.c
_Bratu3D.c: Bratu3D.i Bratu3DApp.h
	${SWIG} -python -I${PETSC4PY_INCLUDE} -o $@ $<


build: _Bratu3D.so
_Bratu3D.so: ${OBJSC} 
	${CLINKER} -shared -o $@ ${OBJSC} ${PETSC_DM_LIB} -L${PYTHON_LIBDIR} -l${PYTHON_LIB}


NP = -n 8
test: build
	${MPIEXEC} ${NP} ${PYTHON} run_Bratu3D.py -snes_monitor -ksp_monitor


clean::
	${RM} _Bratu3D.so Bratu3D.py* _Bratu3D.c ${OBJSC}

#
