-include ../../../../../petscdir.mk
#requirespackage  'PETSC_HAVE_P4EST'

CFLAGS   = ${P4EST_INCLUDE}
SOURCEC  = dmp4est.c dmp8est.c petsc_p4est_package.c
DIRS     = tests
LIBBASE  = libpetscdm
MANSEC   = DM
SUBMANSEC= DMForest

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

