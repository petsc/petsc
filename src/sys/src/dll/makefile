# $Id: makefile,v 1.11 2001/08/22 17:59:22 balay Exp $ 

ALL: lib 
CFLAGS    = -DPETSC_BOPT='"${BOPT}"' -DPETSC_LIB_DIR='"${INSTALL_DIR}/lib/lib${BOPT}/${PETSC_ARCH}"' -DPETSC_SLSUFFIX='"${SLSUFFIX}"'
FFLAGS    =
CPPFLAGS  = 
SOURCEC	  =  dl.c   reg.c
SOURCEF	  =
SOURCEH	  = 
OBJSC	  = dl.o   reg.o
OBJSF	  =
MANSEC	  = Sys
LIBBASE	  = libpetsc
DIRS	  = 
LOCDIR	  = src/sys/src/dll/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test


