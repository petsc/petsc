# $Id: makefile,v 1.26 1998/03/19 02:04:59 balay Exp balay $

ALL: lib

CFLAGS   = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"'
SOURCEC  =  
SOURCEF  =
SOURCEH  = dvecimpl.h
OBJSC    =  
OBJSF    =
LIBBASE  = libpetscvec
DIRS     = seq mpi shared
LOCDIR   = src/vec/impls/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

