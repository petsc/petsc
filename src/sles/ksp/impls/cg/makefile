# $Id: makefile,v 1.30 1998/03/19 02:05:27 balay Exp balay $

ALL: lib 

CFLAGS   = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"' 
FFLAGS   =
SOURCEC  = cg.c cgeig.c cgtype.c
SOURCEF  =
SOURCEH  = cgctx.h 
OBJSC    = cg.o cgeig.o cgtype.o
OBJSF    =
LIBBASE  = libpetscsles
MANSEC   = 5
LOCDIR   = src/ksp/impls/cg/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
