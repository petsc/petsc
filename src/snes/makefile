# $Id: makefile,v 1.18 1997/08/22 15:17:47 bsmith Exp balay $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} ${CONF} -D__SDIR__='"${LOCDIR}"'
SOURCEC  =
SOURCEF  =
SOURCEH  = snesimpl.h
OBJSC    =
OBJSF    =
LIBBASE  = libpetscsnes
DIRS     = interface impls  examples
LOCDIR   = src/snes/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

