# $Id: makefile,v 1.23 1998/03/19 02:06:03 balay Exp balay $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"'
SOURCEC  =
SOURCEF  =
SOURCEH  = pcimpl.h
OBJSC    =
OBJSF    =
LIBBASE  = libpetscsles
DIRS     = interface impls  examples
LOCDIR   = src/pc/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
