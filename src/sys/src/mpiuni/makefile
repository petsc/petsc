# $Id: makefile,v 1.16 1998/03/19 02:14:32 balay Exp balay $ 

ALL: lib

CFLAGS	 = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"'
SOURCEC	 = mpi.c
SOURCEF	 = 
SOURCEH	 = mpi.h
OBJSC	 = mpi.o
OBJSF	 =
LIBBASE	 = libmpiuni
LOCDIR   = src/mpiuni/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
