# $Id: makefile,v 1.4 2001/09/26 17:10:13 balay Exp $

#requirespackage  'PETSC_HAVE_ESI'
#requireslanguage  CPP
#requiresscalar    real 
#requiresprecision double

ALL: lib

CFLAGS   = ${ESI_INCLUDE}  ${CCA_INCLUDE} ${TRILINOS_INCLUDE}
FFLAGS   =
SOURCEC  = eindexspace.c evector.c
OBJSC    = eindexspace.o evector.o
SOURCEH  = ../../../include/esi/petsc/indexspace.h ../../../include/esi/petsc/vector.h
LIBBASE  = libpetscvec
MANSEC   = 
LOCDIR   = src/vec/esi/


include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test


