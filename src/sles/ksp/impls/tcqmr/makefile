# $Id: makefile,v 1.28 1998/03/19 02:05:38 balay Exp balay $ 

ALL: lib 

CFLAGS   = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"'
SOURCEC  = tcqmr.c
SOURCEF  =
SOURCEH  = tcqmrp.h
OBJSC    = tcqmr.o
OBJSF    = 
LIBBASE  = libpetscsles
LOCDIR   = src/ksp/impls/tcqmr/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
