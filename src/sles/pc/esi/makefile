# $Id: makefile,v 1.2 2001/09/19 16:09:57 bsmith Exp bsmith $

#requirespackage  'PETSC_HAVE_ESI'
#requireslanguage  CPP
#requiresscalar    real 
#requiresprecision double

ALL: lib

CFLAGS   = ${ESI_INCLUDE}  ${CCA_INCLUDE}
FFLAGS   =
SOURCEC  = epreconditioner.c
OBJSC    = epreconditioner.o
SOURCEH  = ../../../../include/esi/petsc/preconditioner.h
LIBBASE  = libpetscsles
MANSEC   = 
LOCDIR   = src/sles/pc/esi/


include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test


