# $Id: makefile,v 1.17 1998/04/23 23:27:03 balay Exp bsmith $ 

ALL: lib

CFLAGS	 = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"'
SOURCEC	 = mpi.c
SOURCEF	 = 
SOURCEH	 = mpi.h
OBJSC	 = mpi.o
OBJSF	 =
LIBBASE	 = libmpiuni
LOCDIR   = src/sys/src/mpiuni/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
