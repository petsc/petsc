#requiresdefine   'PETSC_HAVE_FORTRAN'
#

CFLAGS	   =
FFLAGS	   =
SOURCEC	   =
SOURCEF90  = petscmatmod.F90
SOURCEH	   = petscmat.h petscmat.h90 petscmatfactorinfosize.h petscmatinfosize.h
LIBBASE	   = libpetscmat
MANSEC	   = Mat
LOCDIR	   = src/mat/f90-mod/
CLEANFILES = *.mod

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
