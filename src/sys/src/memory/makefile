# $Id: makefile,v 1.2 1998/05/08 16:12:48 bsmith Exp bsmith $ 

ALL: lib 
CFLAGS	  = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"' -DPETSC_BOPT='"${BOPT}"' \
            -DPETSC_LDIR='"${LDIR}"' -DPETSC_DIR='"${PETSC_DIR}"' -DPETSC_SLSUFFIX='"${SLSUFFIX}"'

SOURCEC	  = mal.c   mem.c   mtr.c
SOURCEF	  =
SOURCEH	  = 
OBJSC	  = mal.o   mem.o   mtr.o
OBJSF	  =
MANSEC	  = 7
LIBBASE	  = libpetscsys
DIRS	  = 
LOCDIR	  = src/sys/src/memory/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base




