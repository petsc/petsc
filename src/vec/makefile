# $Id: makefile,v 1.19 1997/08/22 15:10:21 bsmith Exp balay $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} ${CONF} -D__SDIR__='"${LOCDIR}"' 
SOURCEC  =
SOURCEF  =
SOURCEH  = vecimpl.h
OBJSC    =
OBJSF    =
LIBBASE  = libpetscvec
DIRS     = utils interface impls  examples
LOCDIR   = src/vec/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

