# $Id: makefile,v 1.43 2001/04/10 19:35:25 bsmith Exp balay $ 

ALL: lib

CFLAGS   = 
FFLAGS   =
SOURCEC	 = mpiaij.c mmaij.c mpiaijpc.c mpiov.c fdmpiaij.c pxxt.c
SOURCEF	 =
SOURCEH	 = mpiaij.h
OBJSC	 = mpiaij.o mmaij.o mpiaijpc.o mpiov.o fdmpiaij.o pxxt.o
OBJSF	 =
LIBBASE	 = libpetscmat
DIRS	 = 
MANSEC	 = Mat
LOCDIR	 = src/mat/impls/aij/mpi/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test
