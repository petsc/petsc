
#requiresdefine  'PETSC_HAVE_MATLAB_ENGINE'

ALL: lib

CFLAGS    = ${MATLAB_INCLUDE}
FFLAGS    =
CPPFLAGS  =
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


