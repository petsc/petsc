# $Id: makefile,v 1.1 2001/08/30 19:37:15 bsmith Exp bsmith $

#requirespackage  'PETSC_HAVE_ESI'
#requireslanguage  CPP
#requiresscalar    real 
#requiresprecision double

ALL: lib

CFLAGS   = ${ESI_INCLUDE}
FFLAGS   =
SOURCEC  = eobject.c
OBJSC    = eobject.o
SOURCEH  = ${PETSC_DIR}/include/esi/petsc/object.h
LIBBASE  = libpetsc
MANSEC   = 
LOCDIR   = src/sys/src/esi


include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test


