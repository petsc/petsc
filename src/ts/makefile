# $Id: makefile,v 1.8 1998/03/19 02:11:01 balay Exp balay $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"'
SOURCEC  =
SOURCEF  =
SOURCEH  = tsimpl.h
OBJSC    =
OBJSF    =
LIBBASE  = libpetscts
DIRS     = interface impls  examples
LOCDIR   = src/ts/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

