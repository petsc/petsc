# $Id: makefile,v 1.17 1998/03/19 02:08:13 balay Exp balay $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"' \
	   ${SUPERLU_INCLUDE}
SOURCEC  = baij.c baij2.c baijfact.c baijfact2.c dgefa.c dgedi.c dgefa3.c \
	   dgefa4.c dgefa5.c
SOURCEF  =
SOURCEH  = baij.h
OBJSC    = baij.o baij2.o baijfact.o baijfact2.o dgefa.o dgedi.o dgefa3.o \
	   dgefa4.o dgefa5.o
OBJSF    =
LIBBASE  = libpetscmat
DIRS     = 
MANSEC   = 2
LOCDIR   = src/mat/impls/baij/seq/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
