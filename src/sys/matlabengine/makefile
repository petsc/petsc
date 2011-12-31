
#requiresdefine  'PETSC_HAVE_MATLAB_ENGINE'
#requiresscalar real
#requiresprecision  double

ALL: lib 

CFLAGS    = ${MATLAB_INCLUDE}
FFLAGS    =
CPPFLAGS  = 
SOURCEC	  = matlab.c
SOURCEF	  =
SOURCEH	  = ../../../include/petscmatlab.h
OBJSC	  = matlab.o
OBJSF	  =
MANSEC	  = Sys
LIBBASE	  = libpetscsys
LOCDIR	  = src/sys/matlabengine/
DIRS      = 

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test


