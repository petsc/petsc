# $Id: makefile,v 1.5 2000/04/09 03:09:16 bsmith Exp bsmith $ 

#requirespackage  '-DPETSC_HAVE_AMS'

ALL: lib 

CFLAGS    =
FFLAGS    =
CPPFLAGS  = 
SOURCEC	  = pams.c aoptions.c
SOURCEF	  =
SOURCEH	  = 
OBJSC	  = pams.o aoptions.o
OBJSF	  =
MANSEC	  = Sys
LIBBASE	  = libpetsc
DIRS	  = java
LOCDIR	  = src/sys/src/objects/ams/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
include ${PETSC_DIR}/bmake/common_test

