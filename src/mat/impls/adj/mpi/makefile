# $Id: makefile,v 1.3 1998/03/19 02:08:35 balay Exp balay $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"'
SOURCEC  = mpiadj.c
SOURCEF  =
SOURCEH  = mpiadj.h
OBJSC    = mpiadj.o
OBJSF    =
LIBBASE  = libpetscmat
DIRS     = 
MANSEC   = 2
LOCDIR   = src/mat/impls/adj/mpi/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
