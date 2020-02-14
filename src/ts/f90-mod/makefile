#requiresdefine   'PETSC_HAVE_FORTRAN'
#

CFLAGS	   =
FFLAGS	   =
SOURCEC	   =
SOURCEF90  = petsctsmod.F90
SOURCEH	   = petscts.h petscts.h90
LIBBASE	   = libpetscts
MANSEC	   = TS
LOCDIR	   = src/ts/f90-mod/
CLEANFILES = *.mod

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
