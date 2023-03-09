-include ../../../petscdir.mk
#requiresdefine   'PETSC_HAVE_FORTRAN'

SOURCEF90  = petsctsmod.F90
SOURCEH	   = petscts.h petscts.h90
LIBBASE	   = libpetscts
MANSEC	   = TS
CLEANFILES = *.mod

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
