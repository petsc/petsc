# $Id: makefile,v 1.5 2000/10/24 20:24:43 bsmith Exp bsmith $ 

#requirespackage  'PETSC_HAVE_MATLAB_ENGINE'
#requiresscalar real 

ALL: lib 

CFLAGS    = ${MATLAB_INCLUDE}
FFLAGS    =
CPPFLAGS  = 
SOURCEC	  = matlab.c
SOURCEF	  =
SOURCEH	  = 
OBJSC	  = matlab.o
OBJSF	  =
MANSEC	  = Sys
LIBBASE	  = libpetsc
LOCDIR	  = src/sys/src/objects/matlabengine/
DIRS      = examples

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
include ${PETSC_DIR}/bmake/common_test


