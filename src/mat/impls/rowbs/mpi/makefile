# $Id: makefile,v 1.31 1999/02/02 21:26:38 balay Exp bsmith $ 

#requirespackage  '-DPETSC_HAVE_BLOCKSOLVE'
#requiresscalar real 

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
