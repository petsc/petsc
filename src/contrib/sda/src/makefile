# $Id: makefile,v 1.8 1997/09/10 18:55:01 balay Exp balay $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} ${CONF} -D__SDIR__='"${LOCDIR}"'
SOURCEC  = sda2.c sda2f.c
SOURCEF  =
SOURCEH  = sda.h
OBJSC    = sda2.o sda2f.o
OBJSF    =
LIBBASE  = libpetsccontrib
DIRS     = 
LOCDIR   = src/contrib/sda/src

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

