# $Id: makefile,v 1.1 2001/09/14 03:18:02 bsmith Exp bsmith $

#requirespackage  'PETSC_HAVE_ESI'
#requireslanguage  CPP
#requiresscalar    real 
#requiresprecision double

ALL: lib

CFLAGS   = ${ESI_INCLUDE}
FFLAGS   =
SOURCEC  = epreconditioner.c
OBJSC    = epreconditioner.o
SOURCEH  = ${PETSC_DIR}/include/esi/petsc/preconditioner.h
LIBBASE  = libpetscsles
MANSEC   = 
LOCDIR   = src/sles/pc/esi/


include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test


