# $Id: makefile,v 1.21 1997/08/22 15:09:53 bsmith Exp balay $ 

ALL: lib

CFLAGS	 = ${CPPFLAGS} ${CONF} -D__SDIR__='"${LOCDIR}"'
SOURCEC	 =
SOURCEF	 =
SOURCEH	 = isimpl.h
OBJSC	 =
OBJSF	 =
LIBBASE	 = libpetscvec
DIRS	 = interface impls  examples utils
LOCDIR   = src/is/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

