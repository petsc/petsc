# $Id: makefile,v 1.2 2001/09/07 20:09:07 bsmith Exp $

#requirespackage  'PETSC_HAVE_ESI'
#requireslanguage  CPP
#requiresscalar    real 
#requiresprecision double

ALL: lib

CFLAGS   = ${ESI_INCLUDE}
FFLAGS   =
SOURCEC  = emap.c evector.c
OBJSC    = emap.o evector.o
SOURCEH  = ${PETSC_DIR}/include/esi/petsc/map.h ${PETSC_DIR}/include/esi/petsc/vector.h
LIBBASE  = libpetscvec
MANSEC   = 
LOCDIR   = src/vec/esi


include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test


