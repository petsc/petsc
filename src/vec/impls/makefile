# $Id: makefile,v 1.25 1997/11/28 16:18:20 bsmith Exp balay $

ALL: lib

CFLAGS   = ${CPPFLAGS} ${CONF} -D__SDIR__='"${LOCDIR}"'
SOURCEC  =  
SOURCEF  =
SOURCEH  = dvecimpl.h
OBJSC    =  
OBJSF    =
LIBBASE  = libpetscvec
DIRS     = seq mpi shared
LOCDIR   = src/vec/impls/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

