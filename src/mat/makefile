# $Id: makefile,v 1.21 1997/08/22 15:13:04 bsmith Exp balay $ 

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
