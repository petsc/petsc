# $Id: makefile,v 1.5 2001/08/22 18:01:04 balay Exp $ 

#requirespackage  'PETSC_HAVE_ADIC'
#requiresscalar real 
#requiresprecision double

ALL: lib

CFLAGS   = 
FFLAGS   =
SOURCEC  = matadic.c nladic.c
SOURCEF  = 
SOURCEH  = 
OBJSC    = matadic.o nladic.o
OBJSF    = 
LIBBASE  = libpetscsnes
DIRS     = 
MANSEC   = Mat
LOCDIR   = src/mat/impls/adic/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test
