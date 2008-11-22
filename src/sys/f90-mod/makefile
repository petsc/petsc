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
SOURCEF	   = petscmod.F
SOURCEH	   = 
OBJSC	   = 
OBJSF	   = petscmod.o
LIBBASE	   = libpetsc
MANSEC	   = Sys
LOCDIR	   = src/sys/f90-mod/
CLEANFILES = *.mod

include ${PETSC_DIR}/conf/base
include ${PETSC_DIR}/conf/test

buildmod:
	-@${OMAKE}  clean
	@${OMAKE}  petscmod.o
	@${OMAKE}  modcopy
	@${OMAKE}  clean
modcopy:
	${CP} -f *.mod ${PETSC_DIR}/${PETSC_ARCH}/include
