# $Id: makefile,v 1.25 1997/09/10 18:50:09 balay Exp balay $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} ${CONF} -D__SDIR__='"${LOCDIR}"' \
	   ${BS_INCLUDE}
SOURCEC  = mpirowbs.c cholbs.c ilubs.c iccbs.c
SOURCEF  =
SOURCEH  = mpirowbs.h
OBJSC    = mpirowbs.o cholbs.o ilubs.o iccbs.o
OBJSF    =
LIBBASE  = libpetscmat
DIRS     =
MANSEC   = 2
LOCDIR   = src/mat/impls/rowbs/mpi/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
