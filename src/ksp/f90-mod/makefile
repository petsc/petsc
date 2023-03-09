-include ../../../petscdir.mk
#requiresdefine   'PETSC_HAVE_FORTRAN'
#

SOURCEF90  = petsckspdefmod.F90 petscpcmod.F90 petsckspmod.F90
SOURCEH	   = petscksp.h petscksp.h90 petscpc.h petscpc.h90
LIBBASE	   = libpetscksp
MANSEC	   = KSP
CLEANFILES = *.mod

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
