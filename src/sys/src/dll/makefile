# $Id: makefile,v 1.8 2000/02/06 15:46:22 bsmith Exp bsmith $ 

ALL: lib 
CFLAGS    = -DPETSC_BOPT='"${BOPT}"' -DPETSC_LDIR='"${LDIR}"' \
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


