-include ../../../../../petscdir.mk
#requirespackage  'PETSC_HAVE_P4EST'

ALL: lib

CPPFLAGS =
CFLAGS   = ${P4EST_INCLUDE}
FFLAGS   =
SOURCEC  = dmp4est.c dmp8est.c petsc_p4est_package.c
SOURCEF  =
SOURCEH  =
DIRS     = tests
LIBBASE  = libpetscdm
MANSEC   = DM
SUBMANSEC= DMFOREST
LOCDIR   = src/dm/impls/forest/p4est/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

