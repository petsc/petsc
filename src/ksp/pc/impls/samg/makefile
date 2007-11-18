
#requirespackage  'PETSC_HAVE_SAMG'
#requiresscalar    real 
#requiresprecision double
#requireslanguage  CXXONLY

ALL: lib
SPECIALFASTLIB = yes
specialfastlib: lib

CFLAGS   =
FFLAGS   = ${SAMG_MOD}
SOURCEC  = samgmgpetsc.c samgpetsc.c samgpetsctools.c 
SOURCEF  = samgtools1.F
SOURCEH  = petscfunc.h samgfunc.h     
OBJSC    = samgmgpetsc.o samgpetsc.o samgpetsctools.o 
OBJSF    = samgtools1.o
DOCS     = readme
LIBBASE  = libpetscksp
MANSEC   = PC
LOCDIR   = src/ksp/pc/impls/samg/

include ${PETSC_DIR}/conf/base
include ${PETSC_DIR}/conf/test
