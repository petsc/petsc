#requirespackage 'PETSC_HAVE_RANDOM123'

ALL: lib

CFLAGS   = ${RANDOM123_INCLUDE}
FFLAGS   =
SOURCEC  = random123.c
SOURCEF  =
SOURCEH  =
LIBBASE  = libpetscsys
DIRS     =
LOCDIR   = src/sys/classes/random/impls/random123/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

