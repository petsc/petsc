# $Id: makefile,v 1.4 2001/08/02 15:56:25 bsmith Exp balay $ 

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

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test
