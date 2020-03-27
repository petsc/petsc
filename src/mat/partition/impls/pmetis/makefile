-include ../petscdir.mk
ALL: lib

#requirespackage 'PETSC_HAVE_PARMETIS'

CFLAGS    =
FFLAGS    =
CPPFLAGS  = ${PARMETIS_INCLUDE}
SOURCEC   = pmetis.c
SOURCEH   =
LIBBASE   = libpetscmat
LOCDIR    = src/mat/partition/impls/pmetis/
MANSEC    = Mat
SUBMANSEC = MatOrderings

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
