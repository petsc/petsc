#requiresdefine   'PETSC_HAVE_FORTRAN'

ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = zdaf.c zda1f.c zda2f.c zda3f.c zdacornf.c zdagetscatterf.c zdaviewf.c
SOURCEF  =
SOURCEH  =
LIBBASE  = libpetscdm
MANSEC   = DM
SUBMANSEC= DMDA
LOCDIR   = src/dm/impls/da/ftn-custom/

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test


