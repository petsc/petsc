# $Id: makefile,v 1.2 2000/02/09 19:38:27 bsmith Exp $ 

ALL: lib

CFLAGS   = 
FFLAGS   =
SOURCEC  = cstring.c
SOURCEF  =
SOURCEH  = 
OBJSC    = cstring.o
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = Vec
LOCDIR   = src/pf/impls/string/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

#
#  This rule is called by the function PFStringCreateFunction() and creates a dynamic
# library with a function in it named FNAME
petscdlib.c:
	sed -e "s/FUNCTION/${STRINGFUNCTION}/g" -e "s/_MIN_/${MIN}/g" -e "s/_NOUT_/${NOUT}/g" ${PETSC_DIR}/src/pf/impls/string/cstringbase.c > petscdlib.c

petscdlib:  petscdlib.c
	-@${RM} libpetscdlib.a libpetscdlib.${SLSUFFIX}
	-@${OMAKE} -f petscmakefile PETSC_ARCH=${PETSC_ARCH} BOPT=${BOPT} LIBNAME=libpetscdlib.a OBJSC=petscdlib.o libc
	-@${OMAKE} -f petscmakefile PETSC_ARCH=${PETSC_ARCH} BOPT=${BOPT} LIBNAME=libpetscdlib oshared
#	-@${RM} petscdlib.c

