# $Id: makefile,v 1.20 1997/08/22 15:11:03 bsmith Exp balay $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} ${CONF} -D__SDIR__='"${LOCDIR}"'
SOURCEC  =
SOURCEF  =
SOURCEH  = kspimpl.h
OBJSC    =
OBJSF    =
LIBBASE  = libpetscvec
DIRS     = interface impls
LOCDIR   = src/ksp/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
