# $Id: makefile,v 1.29 1998/06/15 16:12:49 balay Exp bsmith $ 

ALL: lib

CFLAGS   =
FFLAGS   =
CPPFLAGS = ${BS_INCLUDE}
SOURCEC  = mpirowbs.c cholbs.c iccbs.c
SOURCEF  =
SOURCEH  = mpirowbs.h
OBJSC    = mpirowbs.o cholbs.o iccbs.o
OBJSF    =
LIBBASE  = libpetscmat
DIRS     =
MANSEC   = 2
LOCDIR   = src/mat/impls/rowbs/mpi/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
