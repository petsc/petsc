# $Id: makefile,v 1.44 2001/08/22 17:59:55 balay Exp $

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

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test

