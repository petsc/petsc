# $Id: makefile,v 1.11 1997/09/10 18:48:59 balay Exp balay $ 

ALL: lib

CFLAGS	 = ${CPPFLAGS} ${CONF} -D__SDIR__='"${LOCDIR}"' 
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
