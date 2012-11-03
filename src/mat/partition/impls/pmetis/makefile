#
ALL: lib

#requirespackage 'PETSC_HAVE_PARMETIS'

CFLAGS   =
FFLAGS   =
CPPFLAGS = ${PARMETIS_INCLUDE}
SOURCEC  = pmetis.c
SOURCEH  = 
LIBBASE  = libpetscmat
LOCDIR   = src/mat/partition/impls/pmetis/
MANSEC   = MatOrderings

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test
