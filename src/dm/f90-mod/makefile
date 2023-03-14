-include ../../../petscdir.mk
#requiresdefine   'PETSC_HAVE_FORTRAN'
#

SOURCEF90  = petscdmmod.F90 petscdmplexmod.F90 petscdmdamod.F90 petscdmswarmmod.F90
LIBBASE	   = libpetscdm
MANSEC	   = DM
CLEANFILES = *.mod

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
