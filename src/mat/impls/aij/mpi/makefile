# $Id: makefile,v 1.44 2001/08/22 18:00:29 balay Exp $ 

ALL: lib

CFLAGS   = ${MUMPS_INCLUDE}
FFLAGS   =
SOURCEC	 = mpiaij.c mmaij.c mpiaijpc.c mpiov.c fdmpiaij.c pxxt.c mumps.c
SOURCEF	 =
SOURCEH	 = mpiaij.h
OBJSC	 = mpiaij.o mmaij.o mpiaijpc.o mpiov.o fdmpiaij.o pxxt.o mumps.o
OBJSF	 =
LIBBASE	 = libpetscmat
DIRS	 = spooles superlu_dist
MANSEC	 = Mat
LOCDIR	 = src/mat/impls/aij/mpi/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test
