# $Id: makefile,v 1.1 2000/04/02 04:46:38 bsmith Exp bsmith $ 

#requirespackage  '-DPETSC_HAVE_MATLAB'
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
LOCDIR	  = src/sys/src/objects/matlabengine
DIRS      = examples

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base


