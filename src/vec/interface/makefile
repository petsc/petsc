# $Id: makefile,v 1.30 1998/04/23 23:30:08 balay Exp bsmith $

ALL: lib

CFLAGS   = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"'
SOURCEC  = vector.c map.c
SOURCEF  =
SOURCEH  = 
OBJSC    = vector.o map.o
OBJSF    =
LIBBASE  = libvec
MANSEC   = 1
LOCDIR   = src/vec/interface/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

