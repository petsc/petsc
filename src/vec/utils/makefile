# $Id: makefile,v 1.45 2001/10/04 17:13:31 balay Exp $

ALL: lib

CFLAGS   = ${MATLAB_INCLUDE} ${PNETCDF_INCLUDE}
FFLAGS   =
SOURCEC  = vinv.c gcreatev.c vscat.c vpscat.c cmesh.c vecio.c \
           pipeline.c comb.c vecstash.c vecmpitoseq.c
SOURCEF  =
SOURCEH  = 
OBJSC    = vinv.o gcreatev.o vscat.o vpscat.o cmesh.o vecio.o \
           pipeline.o comb.o vecstash.o vecmpitoseq.o
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = Vec
LOCDIR   = src/vec/utils/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test

