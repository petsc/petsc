# $Id: makefile,v 1.19 1997/08/22 15:15:56 bsmith Exp balay $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} ${CONF} -D__SDIR__='"${LOCDIR}"'
SOURCEC  =
SOURCEF  =
SOURCEH  = drawimpl.h
OBJSC    =
OBJSF    =
LIBBASE  = libpetscsys
DIRS     = interface impls examples utils
LOCDIR   = src/draw/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
