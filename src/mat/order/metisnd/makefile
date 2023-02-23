-include ../../../../petscdir.mk
#requirespackage   'PETSC_HAVE_METIS'

CFLAGS    = ${METIS_INCLUDE}
SOURCEC   = metisnd.c
SOURCEF   =
SOURCEH   =
LIBBASE   = libpetscmat
DIRS      =
MANSEC    = Mat
SUBMANSEC = MatOrderings

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
