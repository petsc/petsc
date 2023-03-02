-include ../../../petscdir.mk
#requiresdefine   'PETSC_HAVE_FORTRAN'

SOURCEF90  = petscsnesmod.F90
SOURCEH	   = petscsnes.h petscsnes.h90 petscsnesfas.h
LIBBASE	   = libpetscsnes
MANSEC	   = SNES
CLEANFILES = *.mod

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
