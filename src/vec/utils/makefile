# $Id: makefile,v 1.32 1998/03/19 02:04:56 balay Exp balay $

ALL: lib

CFLAGS   = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"' 
SOURCEC  = vinv.c gcreatev.c vscat.c vpscat.c cmesh.c vecio.c
SOURCEF  =
SOURCEH  = 
OBJSC    = vinv.o gcreatev.o vscat.o vpscat.o cmesh.o vecio.o
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = 1
LOCDIR   = src/vec/utils/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

