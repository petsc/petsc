# $Id: makefile,v 1.20 1998/03/19 02:10:09 balay Exp balay $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"'
SOURCEC  =
SOURCEF  =
SOURCEH  = drawimpl.h
OBJSC    =
OBJSF    =
LIBBASE  = libpetscsys
DIRS     = interface impls examples utils
LOCDIR   = src/draw/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
