# $Id: makefile,v 1.22 1997/08/22 15:12:23 bsmith Exp balay $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} ${CONF} -D__SDIR__='"${LOCDIR}"'
SOURCEC  =
SOURCEF  =
SOURCEH  = pcimpl.h
OBJSC    =
OBJSF    =
LIBBASE  = libpetscsles
DIRS     = interface impls  examples
LOCDIR   = src/pc/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
