-include ../../../petsckdir.mk
#requiresdefine   'PETSC_HAVE_FORTRAN'

SOURCEC	   =
SOURCEF90  = petsctaomod.F90
SOURCEH	   = petsctao.h petsctao.h90
LIBBASE	   = libpetsctao
MANSEC	   = Tao
LOCDIR	   = src/tao/f90-mod/
CLEANFILES = *.mod

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
