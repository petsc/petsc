-include ../../../petsckdir.mk
#requiresdefine   'PETSC_HAVE_FORTRAN'

SOURCEC	   =
SOURCEF90  = petscsnesmod.F90
SOURCEH	   = petscsnes.h petscsnes.h90 petscsnesfas.h
LIBBASE	   = libpetscsnes
MANSEC	   = SNES
LOCDIR	   = src/snes/f90-mod/
CLEANFILES = *.mod

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
