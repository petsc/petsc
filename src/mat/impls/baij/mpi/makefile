# $Id: makefile,v 1.12 1998/03/19 02:08:18 balay Exp balay $ 

ALL: lib

CFLAGS	 = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"' 
SOURCEC	 = mpibaij.c mmbaij.c mpibaijpc.c baijpc.c baijov.c
SOURCEF	 =
SOURCEH	 = mpibaij.h
OBJSC	 = mpibaij.o mmbaij.o mpibaijpc.o baijpc.o baijov.o
OBJSF	 =
LIBBASE	 = libpetscmat
DIRS	 = 
MANSEC	 = 2
LOCDIR	 = src/mat/impls/baij/mpi/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
