# $Id: makefile,v 1.19 1998/03/19 02:13:16 balay Exp balay $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"'
SOURCEC  = umls.c search.c
SOURCEF  = 
SOURCEH  = umls.h
OBJSC    = umls.o search.o
OBJSF    = 
LIBBASE  = libpetscsnes
MANSEC   = n
LOCDIR   = src/snes/impls/umls/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

