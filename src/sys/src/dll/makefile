# $Id: makefile,v 1.6 1999/01/12 23:14:11 bsmith Exp balay $ 

ALL: lib 
CFLAGS    =
FFLAGS    =
CPPFLAGS  = -DPETSC_BOPT='"${BOPT}"' -DPETSC_LDIR='"${LDIR}"' \
	-DPETSC_DIR='"${PETSC_DIR}"' -DPETSC_SLSUFFIX='"${SLSUFFIX}"'
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



