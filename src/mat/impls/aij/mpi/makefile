# $Id: makefile,v 1.32 1997/09/10 18:47:16 balay Exp balay $ 

ALL: lib

CFLAGS	 = ${CPPFLAGS} ${CONF} -D__SDIR__='"${LOCDIR}"' 
SOURCEC	 = mpiaij.c mmaij.c mpiaijpc.c aijpc.c mpiov.c fdmpiaij.c
SOURCEF	 =
SOURCEH	 = mpiaij.h
OBJSC	 = mpiaij.o mmaij.o mpiaijpc.o aijpc.o mpiov.o fdmpiaij.o
OBJSF	 =
LIBBASE	 = libpetscmat
DIRS	 = 
MANSEC	 = 2
LOCDIR	 = src/mat/impls/aij/mpi/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
