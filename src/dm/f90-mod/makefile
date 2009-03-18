#requirespackage   'PETSC_USING_F90'
#
#   Makes Fortran module functions from *.h90 files in include/finclude
#
ALL: buildmod
speciallib: buildmod
specialfastlib: buildmod
SPECIALLIB     = yes
SPECIALFASTLIB = yes


CFLAGS	   = 
FFLAGS	   = 
SOURCEC	   = 
SOURCEF	   = petscdmmod.F
SOURCEH	   = 
OBJSC	   = 
OBJSF	   = petscdmmod.o
LIBBASE	   = libpetscdm
MANSEC	   = DA
LOCDIR	   = src/dm/f90-mod/
CLEANFILES = *.mod

include ${PETSC_DIR}/conf/base
include ${PETSC_DIR}/conf/test
