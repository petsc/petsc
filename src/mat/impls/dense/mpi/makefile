# $Id: makefile,v 1.13 1997/09/10 18:47:35 balay Exp balay $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} ${CONF} -D__SDIR__='"${LOCDIR}"'
SOURCEC  = mpidense.c mmdense.c
SOURCEF  =
SOURCEH  = mpidense.h
OBJSC    = mpidense.o mmdense.o
OBJSF    =
LIBBASE  = libpetscmat
DIRS     = 
MANSEC   = 2
LOCDIR   = src/mat/impls/dense/mpi/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
