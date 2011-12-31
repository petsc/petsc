#requiresdefine   'PETSC_USING_F90'
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
SOURCEF	   = mpiunimod.F
SOURCEH	   = 
OBJSC	   = 
OBJSF	   = mpiunimod.o
LIBBASE	   = libpetscsys
MANSEC	   = Sys
LOCDIR	   = src/sys/mpiuni/f90-mod/
CLEANFILES = *.mod

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test
