# $Id: makefile,v 1.9 1998/03/19 02:14:47 balay Exp balay $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"'
SOURCEC  = sda2.c sda2f.c
SOURCEF  =
SOURCEH  = sda.h
OBJSC    = sda2.o sda2f.o
OBJSF    =
LIBBASE  = libpetsccontrib
DIRS     = 
LOCDIR   = src/contrib/sda/src

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

