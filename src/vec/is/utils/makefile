# $Id: makefile,v 1.11 1997/11/03 04:42:34 bsmith Exp balay $

ALL: lib

CFLAGS	 = ${CPPFLAGS} ${CONF} -D__SDIR__='"${LOCDIR}"'
SOURCEC	 = iscomp.c iscoloring.c isltog.c isdiff.c
SOURCEF	 =
SOURCEH	 =
OBJSC	 = iscomp.o iscoloring.o isltog.o isdiff.o
OBJSF	 =
LIBBASE	 = libpetscvec
DIRS	 = 
MANSEC	 = 6
LOCDIR	 = src/is/utils/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
