# $Id: makefile,v 1.2 1999/12/10 01:16:41 knepley Exp $

ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = dense.c
SOURCEF  =
SOURCEH  = ${SOURCEC:.c=.h}
OBJSC    = ${SOURCEC:.c=.o}
OBJSF    =
LIBBASE  = libpetscbilinear
MANSEC   = Bilinear
LOCDIR   = src/bilinear/impls/dense/seq/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test
