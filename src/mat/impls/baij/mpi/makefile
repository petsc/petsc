# $Id: makefile,v 1.19 2000/08/17 04:51:42 bsmith Exp balay $ 

ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC	 = mpibaij.c mmbaij.c baijov.c
SOURCEF	 =
SOURCEH	 = mpibaij.h
OBJSC	 = mpibaij.o mmbaij.o baijov.o
OBJSF	 =
LIBBASE	 = libpetscmat
DIRS	 = 
MANSEC	 = Mat
LOCDIR	 = src/mat/impls/baij/mpi/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test
