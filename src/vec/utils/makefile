# $Id: makefile,v 1.41 1999/03/16 21:54:47 balay Exp bsmith $

ALL: lib

CFLAGS   = ${MATLAB_INCLUDE}
FFLAGS   =
SOURCEC  = vinv.c gcreatev.c vscat.c vpscat.c cmesh.c vecio.c vecreg.c dlregis.c \
           pipeline.c comb.c vecstash.c
SOURCEF  =
SOURCEH  = 
OBJSC    = vinv.o gcreatev.o vscat.o vpscat.o cmesh.o vecio.o vecreg.o dlregis.o \
           pipeline.o comb.o vecstash.o
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = Vec
LOCDIR   = src/vec/utils/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

