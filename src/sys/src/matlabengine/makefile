# $Id: makefile,v 1.6 2001/02/25 04:38:55 bsmith Exp balay $ 

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

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test


