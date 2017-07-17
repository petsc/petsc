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
SOURCEF	   = petscdmmod.F
SOURCEH = petscdmcomposite.h90 petscdmda.h90 petscdmforest.h90          \
 petscdm.h90 petscdmlabel.h90 petscdmnetwork.h90 petscdmpatch.h90       \
 petscdmplex.h90 petscdt.h90
LIBBASE	   = libpetscdm
MANSEC	   = DM
LOCDIR	   = src/dm/f90-mod/
CLEANFILES = *.mod

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
