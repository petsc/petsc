#requiresdefine   'PETSC_HAVE_FORTRAN'
#requiresdefine   'PETSC_USING_F90'
#

CFLAGS	   =
FFLAGS	   =
SOURCEC	   =
SOURCEF	   = petsckspdefmod.F petscpcmod.F petsckspmod.F
SOURCEH	   = petscksp.h petscksp.h90 petscpc.h petscpc.h90
LIBBASE	   = libpetscksp
MANSEC	   = KSP
LOCDIR	   = src/ksp/f90-mod/
CLEANFILES = *.mod

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
