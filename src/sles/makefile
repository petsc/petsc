# $Id: makefile,v 1.20 1998/03/19 02:09:33 balay Exp balay $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"'
SOURCEC  =
SOURCEF  =
SOURCEH  = slesimpl.h
OBJSC    =
OBJSF    =
LIBBASE  = libpetscvec
DIRS     = interface examples
LOCDIR   = src/sles/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
