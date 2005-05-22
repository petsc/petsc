
#requirespackage  'PETSC_HAVE_MATLAB'
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
LIBBASE	  = libpetsc
LOCDIR	  = src/sys/objects/matlabengine/
DIRS      = 

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test


