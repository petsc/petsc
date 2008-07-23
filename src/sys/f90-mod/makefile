#requirespackage   'PETSC_USING_F90'

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
LOCDIR	   = src/fortran/f90mod/
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
