# $Id: makefile,v 1.3 2001/07/27 15:56:51 bsmith Exp bsmith $ 

#requirespackage  'PETSC_HAVE_ADIC'
#requiresscalar real 

ALL: lib

CFLAGS   = ${ADIC_INCLUDE}
FFLAGS   =
SOURCEC  = matadic.c
SOURCEF  = 
SOURCEH  = 
OBJSC    = matadic.o
OBJSF    = 
LIBBASE  = libpetscsnes
DIRS     = 
MANSEC   = Mat
LOCDIR   = src/mat/impls/adic

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
include ${PETSC_DIR}/bmake/common_test
