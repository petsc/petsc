# $Id: makefile,v 1.44 2001/08/22 18:00:29 balay Exp $ 

ALL: lib

CFLAGS   = ${SUPERLUDIST_INCLUDE} ${SPOOLES_INCLUDE}
FFLAGS   =
SOURCEC	 = mpiaij.c mmaij.c mpiaijpc.c mpiov.c fdmpiaij.c pxxt.c superlu_dist.c mpispooles.c
SOURCEF	 =
SOURCEH	 = mpiaij.h
OBJSC	 = mpiaij.o mmaij.o mpiaijpc.o mpiov.o fdmpiaij.o pxxt.o superlu_dist.o mpispooles.o
OBJSF	 =
LIBBASE	 = libpetscmat
DIRS	 = 
MANSEC	 = Mat
LOCDIR	 = src/mat/impls/aij/mpi/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test
