# $Id: makefile,v 1.7 2001/08/22 17:59:41 balay Exp $ 

#requirespackage  'PETSC_HAVE_MATLAB_ENGINE'
#requiresscalar real 

ALL: lib 

CFLAGS    = ${MATLAB_ENGINE_INCLUDE}
FFLAGS    =
CPPFLAGS  = 
SOURCEC	  = matlab.c
SOURCEF	  =
SOURCEH	  = ../../../../include/petscengine.h
OBJSC	  = matlab.o
OBJSF	  =
MANSEC	  = Sys
LIBBASE	  = libpetsc
LOCDIR	  = src/sys/src/objects/matlabengine/
DIRS      = examples

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test


