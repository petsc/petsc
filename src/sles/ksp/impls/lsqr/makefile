# $Id: makefile,v 1.26 1998/03/19 02:05:35 balay Exp balay $ 

ALL: lib 

CFLAGS   = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"' 
SOURCEC  = lsqr.c
SOURCEF  =
OBJSC    = lsqr.o
OBJSF    =
LIBBASE  = libpetscsles
LOCDIR   = src/ksp/impls/lsqr/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
