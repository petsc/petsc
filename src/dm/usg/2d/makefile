# $Id: makefile,v 1.1 1998/04/03 20:33:06 bsmith Exp bsmith $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} ${CONF} -D__SDIR__='"${LOCDIR}"'
SOURCEC  =
SOURCEF  =
SOURCEH  = matimpl.h
OBJSC    =
OBJSF    =
LIBBASE  = libpetscmat
DIRS     = interface impls examples utils
LOCDIR   = src/mat/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
