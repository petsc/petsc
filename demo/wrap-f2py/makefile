.PHONY: source build test clean

MODULE = Bratu2D

ALL: build
source: ${MODULE}module.c
build:  ${MODULE}.so

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

PYTHON = python
F2PY   = f2py

#

PYTHON_INCLUDE = ${shell ${PYTHON} -c 'from distutils import sysconfig; print(sysconfig.get_python_inc())'}
PYTHON_LIBDIR  = ${shell ${PYTHON} -c 'from distutils import sysconfig; print(sysconfig.get_config_var("LIBDIR"))'}
PYTHON_LIB     = ${shell ${PYTHON} -c 'from distutils import sysconfig; print("python"+sysconfig.get_python_version())'}
NUMPY_INCLUDE  = ${shell ${PYTHON} -c 'import numpy; print(numpy.get_include())'}
F2PY_ROOTDIR   = ${shell ${PYTHON} -c 'import os, numpy.f2py; print(os.path.dirname(numpy.f2py.__file__))'}

#

F2PY_FLAGS =
fortranobject.h:
	cp ${F2PY_ROOTDIR}/src/$@ ./$@
fortranobject.c: fortranobject.h
	cp ${F2PY_ROOTDIR}/src/$@ ./$@
${MODULE}module.c: ${MODULE}.pyf fortranobject.c
	${F2PY} ${F2PY_FLAGS}  $<
#

CFLAGS  = -I${PYTHON_INCLUDE} -I${NUMPY_INCLUDE}
SOURCEC = ${MODULE}module.c fortranobject.c
OBJSC   = ${MODULE}module.o fortranobject.o

FFLAGS  = ${FC_FLAGS} ${FCPPFLAGS}
SOURCEF = ${MODULE}.F90
OBJSF   = ${MODULE}.o

${MODULE}.so: ${OBJSC} ${OBJSF}
	${FLINKER} -shared ${PETSC_DM_LIB} -L${PYTHON_LIBDIR} -l${PYTHON_LIB} -o $@ ${OBJSC} ${OBJSF}

#

NP = -n 8
test: build
	${MPIEXEC} ${NP} ${PYTHON} run_demo.py -snes_monitor -ksp_monitor

#

clean::
	${RM} fortranobject.[ch] ${MODULE}module.c ${OBJSC} ${OBJSF} ${MODULE}.so
	${RM} -r build
#
