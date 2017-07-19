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
SOURCEF	   = petscsysmod.F
SOURCEH	   = petscbag.h petscdraw.h petscerror.h petsclog.h petscsys.h  petscviewer.h petscviewer.h90 petscsys.h90
LIBBASE	   = libpetscsys
MANSEC	   = Sys
LOCDIR	   = src/sys/f90-mod/
CLEANFILES = *.mod

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
