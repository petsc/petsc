# $Id: makefile,v 1.23 1999/03/17 23:23:13 bsmith Exp bsmith $ 

ALL: lib

CFLAGS   =
FFLAGS   =
CPPFLAGS = ${SUPERLU_INCLUDE}
SOURCEC  = baij.c baij2.c baijfact.c baijfact2.c dgefa.c dgedi.c dgefa3.c \
	   dgefa4.c dgefa5.c dgefa2.c dgefa6.c dgefa7.c aijbaij.c
SOURCEF  =
SOURCEH  = baij.h
OBJSC    = baij.o baij2.o baijfact.o baijfact2.o dgefa.o dgedi.o dgefa3.o \
	   dgefa4.o dgefa5.o dgefa2.o dgefa6.o dgefa7.o aijbaij.o
OBJSF    =
LIBBASE  = libpetscmat
DIRS     = 
MANSEC   = Mat
LOCDIR   = src/mat/impls/baij/seq/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
