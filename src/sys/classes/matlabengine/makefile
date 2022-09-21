-include ../../../../petscdir.mk
#requiresdefine  'PETSC_HAVE_MATLAB'

CFLAGS    = ${MATLAB_INCLUDE}
SOURCEC	  = matlab.c
SOURCEF	  =
SOURCEH	  = ../../../../include/petscmatlab.h
MANSEC	  = Sys
LIBBASE	  = libpetscsys
LOCDIR	  = src/sys/classes/matlabengine/
DIRS      =

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test


