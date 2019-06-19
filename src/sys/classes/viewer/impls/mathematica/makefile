
#requirespackage  'PETSC_HAVE_MATHEMATICA'
#requiresscalar real

ALL: lib

CFLAGS     = ${MATHEMATICA_INCLUDE}
SOURCEC    = mathematica.c
SOURCEH    = mathematica.h
MANSEC     = Sys
SUBMANSEC  = Viewer
LIBBASE    = libpetscsys
DIRS       =
LOCDIR     = src/sys/classes/viewer/impls/mathematica/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

runtime: runtime.o
	${CLINKER} -o $@ $@.o  ${PETSC_KSP_LIB}
	${RM} -f $@.o

include ${PETSC_DIR}/lib/petsc/conf/test

run: runtime
	./runtime
