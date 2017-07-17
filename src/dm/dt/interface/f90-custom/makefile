#requiresdefine   'PETSC_HAVE_FORTRAN'
#requiresdefine   'PETSC_USING_F90'
ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = zdtf90.c zdtdsf90.c
SOURCEF  =
SOURCEH  =
LIBBASE  = libpetscdm
LOCDIR   = src/dm/dt/interface/f90-custom/


include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
