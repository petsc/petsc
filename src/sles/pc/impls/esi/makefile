# $Id: makefile,v 1.33 2001/08/22 18:01:21 balay Exp $ 

#requireslanguage  CPP
#requiresscalar    real 
#requiresprecision double
#requirespackage   'PETSC_HAVE_CXX_NAMESPACE'

ALL: lib

CFLAGS   = ${ESI_INCLUDE}  ${CCA_INCLUDE}
FFLAGS   =
SOURCEC  = pesi.c
SOURCEF  =
SOURCEH  = 
OBJSC    = pesi.o
OBJSF    =
LIBBASE  = libpetscsles
DIRS     = 
MANSEC   = PC
LOCDIR   = src/sles/pc/impls/esi/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test
