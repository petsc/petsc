# $Id: makefile,v 1.34 1998/05/29 20:35:25 bsmith Exp bsmith $

ALL: lib

CFLAGS   = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"' 
SOURCEC  = vinv.c gcreatev.c vscat.c vpscat.c cmesh.c vecio.c vecreg.c dlregis.c
SOURCEF  =
SOURCEH  = 
OBJSC    = vinv.o gcreatev.o vscat.o vpscat.o cmesh.o vecio.o vecreg.o dlregis.o
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = 1
LOCDIR   = src/vec/utils/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

