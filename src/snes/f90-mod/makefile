#requiresdefine   'PETSC_HAVE_FORTRAN'
#requiresdefine   'PETSC_USING_F90'
#
#   Makes Fortran module functions from *.h90 files in include/petsc/finclude
#
ALL: buildmod
speciallib: buildmod
specialfastlib: buildmodfast
SPECIALLIB     = yes
SPECIALFASTLIB = yes


CFLAGS	   =
FFLAGS	   =
SOURCEC	   =
SOURCEF	   = petscsnesmod.F
SOURCEH	   = petscsnes.h petscsnes.h90 petscsnesfas.h
LIBBASE	   = libpetscsnes
MANSEC	   = SNES
LOCDIR	   = src/snes/f90-mod/
CLEANFILES = *.mod

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
