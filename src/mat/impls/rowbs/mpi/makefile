# $Id: makefile,v 1.32 1999/11/05 14:45:25 bsmith Exp bsmith $ 

#requirespackage  '-DPETSC_HAVE_BLOCKSOLVE'
#requiresscalar real 

ALL: lib

CFLAGS   =
FFLAGS   =
CPPFLAGS = ${BLOCKSOLVE_INCLUDE}
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
