# $Id: makefile,v 1.4 2001/08/22 18:02:49 balay Exp $ 

#requirespackage  'PETSC_HAVE_SAMG'
#requireslanguage  CONLY
#requiresscalar    real 
#requiresprecision double

ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = samgmgpetsc.c samgpetsc.c samgpetsctools.c 
SOURCEF  = samgtools1.F
SOURCEH  = petscfunc.h samgfunc.h     
OBJSC    = samgmgpetsc.o samgpetsc.o samgpetsctools.o 
OBJSF    = samgtools1.o
DOCS     = readme
LIBBASE  = libpetscsles
MANSEC   = PC
LOCDIR   = src/contrib/lahaye/samg/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test
