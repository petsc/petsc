# $Id: makefile,v 1.19 1998/03/19 02:12:47 balay Exp balay $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"'
SOURCEC  =
SOURCEF  =
SOURCEH  = snesimpl.h
OBJSC    =
OBJSF    =
LIBBASE  = libpetscsnes
DIRS     = interface impls  examples
LOCDIR   = src/snes/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

