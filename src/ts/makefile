# $Id: makefile,v 1.7 1997/08/22 15:16:34 bsmith Exp balay $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} ${CONF} -D__SDIR__='"${LOCDIR}"'
SOURCEC  =
SOURCEF  =
SOURCEH  = tsimpl.h
OBJSC    =
OBJSF    =
LIBBASE  = libpetscts
DIRS     = interface impls  examples
LOCDIR   = src/ts/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

