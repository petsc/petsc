# $Id: makefile,v 1.20 2001/08/22 18:00:45 balay Exp $ 

ALL: lib

CFLAGS   = ${DSCPACK_INCLUDE} -DF_NEEDS_UNDSC -DDBL_R_NUM
FFLAGS   =
SOURCEC	 = mpibaij.c mmbaij.c baijov.c dscpack.c
SOURCEF	 =
SOURCEH	 = mpibaij.h
OBJSC	 = mpibaij.o mmbaij.o baijov.o dscpack.o
OBJSF	 =
LIBBASE	 = libpetscmat
DIRS	 = 
MANSEC	 = Mat
LOCDIR	 = src/mat/impls/baij/mpi/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test
