# $Id: makefile,v 1.22 1998/03/19 02:06:59 balay Exp $ 

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
