# $Id: makefile,v 1.19 1997/08/22 15:15:27 bsmith Exp balay $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} ${CONF} -D__SDIR__='"${LOCDIR}"'
SOURCEC  =
SOURCEF  =
SOURCEH  = slesimpl.h
OBJSC    =
OBJSF    =
LIBBASE  = libpetscvec
DIRS     = interface examples
LOCDIR   = src/sles/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
