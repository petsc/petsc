# $Id: makefile,v 1.16 1998/03/19 02:14:04 balay Exp balay $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"'
SOURCEC  =
SOURCEF  =
SOURCEH  = daimpl.h
OBJSC    =
OBJSF    =
LIBBASE  = libpetscvec
DIRS     = src  examples utils
LOCDIR   = src/da/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

