# $Id: makefile,v 1.37 1999/02/02 21:29:21 balay Exp bsmith $ 

ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = gmres.c borthog.c borthog2.c borthog3.c gmres2.c gmreig.c \
           gmpre.c
SOURCEH  = gmresp.h
SOURCEF  = 
OBJSC    = gmres.o borthog.o borthog2.o borthog3.o gmres2.o gmreig.o \
           gmpre.o
OBJSF    = 
LIBBASE  = libpetscsles
MANSEC   = KSP
LOCDIR   = src/sles/ksp/impls/gmres/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
