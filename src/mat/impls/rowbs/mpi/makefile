# $Id: makefile,v 1.28 1998/06/12 21:46:41 balay Exp balay $ 

ALL: lib

CFLAGS   =
FFLAGS   =
CPPFLAGS = ${BS_INCLUDE}
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
