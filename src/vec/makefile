# $Id: makefile,v 1.20 1998/03/19 02:04:53 balay Exp balay $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"' 
SOURCEC  =
SOURCEF  =
SOURCEH  = vecimpl.h
OBJSC    =
OBJSF    =
LIBBASE  = libpetscvec
DIRS     = utils interface impls  examples
LOCDIR   = src/vec/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

