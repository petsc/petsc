-include ../../../../../petscdir.mk
#requirespackage 'PETSC_HAVE_CHACO'

CPPFLAGS  = ${CHACO_INCLUDE}
SOURCEC   = chaco.c
SOURCEH   =
LIBBASE   = libpetscmat
MANSEC    = Mat
SUBMANSEC = MatOrderings

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
