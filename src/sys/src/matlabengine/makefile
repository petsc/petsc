# $Id: makefile,v 1.4 2000/08/17 04:50:52 bsmith Exp bsmith $ 

#requirespackage  'PETSC_HAVE_MATLAB'
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


