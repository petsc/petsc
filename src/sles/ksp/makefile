# $Id: makefile,v 1.21 1998/03/19 02:05:19 balay Exp balay $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"'
SOURCEC  =
SOURCEF  =
SOURCEH  = kspimpl.h
OBJSC    =
OBJSF    =
LIBBASE  = libpetscvec
DIRS     = interface impls
LOCDIR   = src/ksp/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
