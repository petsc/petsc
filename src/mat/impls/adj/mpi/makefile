# $Id: makefile,v 1.2 1997/09/26 02:19:39 bsmith Exp balay $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} ${CONF} -D__SDIR__='"${LOCDIR}"'
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
