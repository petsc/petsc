# $Id: makefile,v 1.9 2000/08/17 04:50:41 bsmith Exp balay $ 

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

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
include ${PETSC_DIR}/bmake/common_test


