# $Id: makefile,v 1.25 1998/03/19 02:07:16 balay Exp balay $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"' 
SOURCEC  = dense.c 
SOURCEF  =
SOURCEH  = dense.h
OBJSC    = dense.o 
OBJSF    =
LIBBASE  = libpetscmat
MANSEC   = 2
LOCDIR   = src/mat/impls/dense/seq/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
