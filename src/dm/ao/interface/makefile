# $Id: makefile,v 1.11 1998/03/19 02:15:20 balay Exp balay $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"'
SOURCEC  = ao.c aodata.c
SOURCEF  = 
SOURCEH  = 
OBJSC    = ao.o aodata.o
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = 6
LOCDIR   = src/ao/interface/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
