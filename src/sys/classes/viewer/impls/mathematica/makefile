-include ../../../../../../petscdir.mk
#requirespackage  'PETSC_HAVE_MATHEMATICA'
#requiresscalar real

CFLAGS     = ${MATHEMATICA_INCLUDE}
MANSEC     = Sys
SUBMANSEC  = Viewer
LIBBASE    = libpetscsys

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules.doc

runtime: runtime.o
	${CLINKER} -o $@ $@.o  ${PETSC_KSP_LIB}
	${RM} -f $@.o


run: runtime
	./runtime
