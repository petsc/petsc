#requiresdefine   'PETSC_HAVE_FORTRAN'

SOURCEC	   =
SOURCEF90  = petscsysmod.F90
SOURCEH	   = petscbag.h petscdraw.h petscerror.h petsclog.h petscsys.h  petscviewer.h petscviewer.h90 petscsys.h90
LIBBASE	   = libpetscsys
MANSEC	   = Sys
LOCDIR	   = src/sys/f90-mod/
CLEANFILES = *.mod

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
