# $Id: makefile,v 1.15 1997/08/22 15:19:02 bsmith Exp balay $ 

ALL: lib

CFLAGS	 = ${CPPFLAGS} ${CONF} -D__SDIR__='"${LOCDIR}"'
SOURCEC	 = mpi.c
SOURCEF	 = 
SOURCEH	 = mpi.h
OBJSC	 = mpi.o
OBJSF	 =
LIBBASE	 = libmpiuni
LOCDIR   = src/mpiuni/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
