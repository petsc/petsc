# $Id: makefile,v 1.6 1997/11/09 04:09:43 bsmith Exp balay $ 

ALL: lib

CFLAGS	 = ${CPPFLAGS} ${CONF} -D__SDIR__='"${LOCDIR}"'
SOURCEC	 =
SOURCEF	 =
SOURCEH	 = aoimpl.h
OBJSC	 =
OBJSF	 =
LIBBASE	 = libpetscvec
DIRS	 = interface impls  examples utils
LOCDIR   = src/is/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

