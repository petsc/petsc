# $Id: makefile,v 1.33 1998/04/23 23:30:56 balay Exp bsmith $

ALL: lib

CFLAGS   = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"' 
SOURCEC  = vinv.c gcreatev.c vscat.c vpscat.c cmesh.c vecio.c
SOURCEF  =
SOURCEH  = 
OBJSC    = vinv.o gcreatev.o vscat.o vpscat.o cmesh.o vecio.o
OBJSF    =
LIBBASE  = libvec
MANSEC   = 1
LOCDIR   = src/vec/utils/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

