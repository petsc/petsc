-include ../../../../petscdir.mk
#requirespackage   'PETSC_HAVE_METIS'

ALL: lib

CFLAGS    = ${METIS_INCLUDE}
FFLAGS    =
SOURCEC   = metisnd.c
SOURCEF   =
SOURCEH   =
LIBBASE   = libpetscmat
DIRS      =
MANSEC    = Mat
SUBMANSEC = MatOrderings
LOCDIR    = src/mat/order/metisnd/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
