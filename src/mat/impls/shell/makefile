# $Id: makefile,v 1.24 1998/03/19 02:07:28 balay Exp balay $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"' 
SOURCEC  = shell.c shellcnv.c
SOURCEF  =
SOURCEH  = 
OBJSC    = shell.o shellcnv.o
OBJSF    =
LIBBASE  = libpetscmat
MANSEC   = 2
LOCDIR   = src/mat/impls/shell/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
