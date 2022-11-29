-include ../../../../../petscdir.mk

SOURCEC  = cstring.c
SOURCEF  =
SOURCEH  =
LIBBASE  = libpetscvec
MANSEC   = Vec
LOCDIR   = src/vec/pf/impls/string/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

#
#  This rule is called by the function PFStringCreateFunction() and creates a dynamic
# library with a function in it named FNAME
petscdlib.c:
	sed -e "s/FUNCTION/${STRINGFUNCTION}/g" -e "s/_MIN_/${MIN}/g" -e "s/_NOUT_/${NOUT}/g" ${PETSC_DIR}/src/pf/impls/string/cstringbase.template > petscdlib.c

petscdlib:  petscdlib.c
	-@${RM} libpetscdlib.a libpetscdlib.${SL_LINKER_SUFFIX}
	-@${OMAKE} PETSC_ARCH=${PETSC_ARCH}  LIBNAME=libpetscdlib.a SOURCEC=petscdlib.c libc
	-@${OMAKE} PETSC_ARCH=${PETSC_ARCH}  LIBNAME=libpetscdlib oshared
#	-@${RM} petscdlib.c

