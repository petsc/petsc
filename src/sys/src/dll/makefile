# $Id: makefile,v 1.3 1998/05/21 14:52:23 bsmith Exp balay $ 

ALL: lib 
CPPFLAGS  = -DPETSC_BOPT='"${BOPT}"' -DPETSC_LDIR='"${LDIR}"' \
	-DPETSC_DIR='"${PETSC_DIR}"' -DPETSC_SLSUFFIX='"${SLSUFFIX}"'
SOURCEC	  =  dl.c   reg.c
SOURCEF	  =
SOURCEH	  = 
OBJSC	  = dl.o   reg.o
OBJSF	  =
MANSEC	  = 7
LIBBASE	  = libpetscsys
DIRS	  = 
LOCDIR	  = src/sys/src/dll/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base



