# $Id: makefile,v 1.7 1998/03/19 02:15:15 balay Exp balay $ 

ALL: lib

CFLAGS	 = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"'
SOURCEC	 =
SOURCEF	 =
SOURCEH	 = aoimpl.h
OBJSC	 =
OBJSF	 =
LIBBASE	 = libpetscvec
DIRS	 = interface impls  examples utils
LOCDIR   = src/is/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

