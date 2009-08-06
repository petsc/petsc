.PHONY: source build test clean

MODULE = Bratu3D

ALL: build
source: _${MODULE}.c
build: _${MODULE}.so

include ${PETSC_DIR}/conf/base

PYTHON = python
SWIG   = swig

#

PYTHON_INCLUDE   = ${shell ${PYTHON} -c 'from distutils import sysconfig; print(sysconfig.get_python_inc())'}
PYTHON_LIBDIR    = ${shell ${PYTHON} -c 'from distutils import sysconfig; print(sysconfig.get_config_var("LIBDIR"))'}
PYTHON_LIB       = ${shell ${PYTHON} -c 'from distutils import sysconfig; print("python"+sysconfig.get_python_version())'}
PETSC4PY_INCLUDE = ${shell ${PYTHON} -c 'import petsc4py; print(petsc4py.get_include())'}

#

CFLAGS  = -I${PYTHON_INCLUDE} -I${PETSC4PY_INCLUDE}
SOURCEC = ${MODULE}_wrap.c ${MODULE}.c
OBJSC   = ${MODULE}_wrap.o ${MODULE}.o

${MODULE}_wrap.c: ${MODULE}.i ${MODULE}.h
	${SWIG} -python -I${PETSC4PY_INCLUDE} -o $@ $<

_${MODULE}.so: ${OBJSC} 
	${CLINKER} -shared -o $@ ${OBJSC} ${PETSC_DM_LIB} -L${PYTHON_LIBDIR} -l${PYTHON_LIB}

#

NP = -n 8
test: build
	${MPIEXEC} ${NP} ${PYTHON} run_demo.py -snes_monitor -ksp_monitor

clean::
	${RM} _${MODULE}.so ${MODULE}.py* ${MODULE}_wrap.c ${OBJSC}
	${RM} -r build
#
