# $Id: makefile,v 1.30 1999/01/26 18:29:38 bsmith Exp balay $ 

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
MANSEC   = Mat
LOCDIR   = src/mat/impls/rowbs/mpi/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
