-include ../../../../../petscdir.mk
#requiresdefine   'PETSC_HAVE_FORTRAN'

SOURCEC  = zdaf.c zda1f.c zda2f.c zda3f.c zdaghostf.c zdacornf.c zdagetscatterf.c zdaviewf.c zdaindexf.c zdasubf.c
SOURCEF  =
SOURCEH  =
LIBBASE  = libpetscdm
MANSEC   = DM
SUBMANSEC= DMDA
LOCDIR   = src/dm/impls/da/ftn-custom/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test


