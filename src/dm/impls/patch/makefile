-include ../petscdir.mk
ALL: lib

CPPFLAGS =
CFLAGS   =
FFLAGS   =
SOURCEC  = patchcreate.c patch.c
SOURCEF  =
SOURCEH  =
DIRS     = tests
LIBBASE  = libpetscdm
MANSEC   = DM
SUBMANSEC= DMPATCH
LOCDIR   = src/dm/impls/patch/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
