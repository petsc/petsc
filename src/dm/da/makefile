# $Id: makefile,v 1.15 1997/08/22 15:18:41 bsmith Exp balay $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} ${CONF} -D__SDIR__='"${LOCDIR}"'
SOURCEC  =
SOURCEF  =
SOURCEH  = daimpl.h
OBJSC    =
OBJSF    =
LIBBASE  = libpetscvec
DIRS     = src  examples utils
LOCDIR   = src/da/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

