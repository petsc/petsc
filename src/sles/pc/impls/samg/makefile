# $Id: makefile,v 1.4 2001/08/22 18:02:49 balay Exp $ 

#requirespackage  'PETSC_HAVE_F90_H'
#requirespackage  'PETSC_HAVE_SAMG'
#requiresscalar    real 
#requiresprecision double
#requireslanguage  CPP

ALL: lib

CFLAGS   =
FFLAGS   = ${SAMG_MOD}
SOURCEC  = samgmgpetsc.c samgpetsc.c samgpetsctools.c 
SOURCEF  = samgtools1.F
SOURCEH  = petscfunc.h samgfunc.h     
OBJSC    = samgmgpetsc.o samgpetsc.o samgpetsctools.o 
OBJSF    = samgtools1.o
DOCS     = readme
LIBBASE  = libpetscsles
MANSEC   = PC
LOCDIR   = src/sles/pc/impls/samg/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test
