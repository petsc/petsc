# $Id: makefile,v 1.10 2001/04/05 21:06:44 balay Exp balay $ 

ALL: lib 
CFLAGS    = -DPETSC_BOPT='"${BOPT}"' -DPETSC_LIB_DIR='"${PETSC_LIB_DIR}"' \
            -DPETSC_DIR='"${PETSC_DIR}"' -DPETSC_SLSUFFIX='"${SLSUFFIX}"'
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


