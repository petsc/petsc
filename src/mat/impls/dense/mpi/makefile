# $Id: makefile,v 1.14 1998/03/19 02:07:18 balay Exp balay $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"'
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
