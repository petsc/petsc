
#requirespackage  'PETSC_HAVE_MATHEMATICA'
#requiresscalar real 

ALL: lib 

CFLAGS    = ${MATHEMATICA_INCLUDE} ${PLAPACK_INCLUDE}
SOURCEC   = mathematica.c
SOURCEH   = mathematica.h
OBJSC     = ${SOURCEC:.c=.o}
MANSEC    = Viewer
LIBBASE   = libpetscsys
DIRS      = 
LOCDIR    = src/sys/viewer/impls/mathematica/

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

runtime: runtime.o chkopts
	${CLINKER} -o $@ $@.o  ${PETSC_KSP_LIB}
	${RM} -f $@.o

include ${PETSC_DIR}/conf/test

run: runtime
	./runtime
