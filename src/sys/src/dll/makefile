# $Id: makefile,v 1.5 1998/06/15 15:41:40 balay Exp bsmith $ 

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
MANSEC	  = 7
LIBBASE	  = libpetsc
DIRS	  = 
LOCDIR	  = src/sys/src/dll/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base



