#requiresdefine   'PETSC_HAVE_FORTRAN'
#requiresdefine   'PETSC_USING_F90'
#

CFLAGS	   =
FFLAGS	   =
SOURCEC	   =
SOURCEF	   = petscmatmod.F
SOURCEH	   = petscmat.h petscmat.h90 petscmatfactorinfosize.h petscmatinfosize.h
LIBBASE	   = libpetscmat
MANSEC	   = Mat
LOCDIR	   = src/mat/f90-mod/
CLEANFILES = *.mod

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
