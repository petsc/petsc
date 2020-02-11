#requiresdefine   'PETSC_HAVE_FORTRAN'
#requiresdefine   'PETSC_USING_F90'
#

CFLAGS	   =
FFLAGS	   =
SOURCEC	   =
SOURCEF90  = petscvecmod.F90
SOURCEH	   = petscao.h petscis.h petscis.h90 petscvec.h petscvec.h90
LIBBASE	   = libpetscvec
MANSEC	   = Vec
LOCDIR	   = src/vec/f90-mod/
CLEANFILES = *.mod

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
