-include ../../../../../petscdir.mk
#requiresdefine  'PETSC_HAVE_POPEN'

LIBBASE  = libpetscvec
MANSEC   = Vec
DIRS     = tests

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

#  These rules are called by the function PFStringCreateFunction() and creates a dynamic library with that function in it named FNAME

petscdlib.c:
	sed -e "s/FUNCTION/${STRINGFUNCTION}/g" -e "s/_NIN_/${NIN}/g" -e "s/_NOUT_/${NOUT}/g" ${PETSC_DIR}/src/vec/pf/impls/string/cstringbase.template > petscdlib.c

petscdlib.o: petscdlib.c

libpetscdlib:  petscdlib.o
	-@${RM} libpetscdlib.${SL_LINKER_SUFFIX}
	${CLINKER} $(call SL_LINKER_FUNCTION,petscdlib,1,0) petscdlib.o -o libpetscdlib.${SL_LINKER_SUFFIX} $(PETSC_EXTERNAL_LIB_BASIC)

