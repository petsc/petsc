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
SOURCEF	   = petscvecmod.F
SOURCEH	   = petscao.h petscis.h petscis.h90 petscvec.h petscvec.h90 
LIBBASE	   = libpetscvec
MANSEC	   = Vec
LOCDIR	   = src/vec/f90-mod/
CLEANFILES = *.mod

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
