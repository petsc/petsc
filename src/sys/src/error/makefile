# $Id: makefile,v 1.2 1998/05/08 16:12:43 bsmith Exp bsmith $ 

ALL: lib 
CFLAGS	  = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"' -DPETSC_BOPT='"${BOPT}"' \
            -DPETSC_LDIR='"${LDIR}"' -DPETSC_DIR='"${PETSC_DIR}"' -DPETSC_SLSUFFIX='"${SLSUFFIX}"'

SOURCEC	  = adebug.c   err.c      fp.c       signal.c   stack.c
SOURCEF	  =
SOURCEH	  = 
OBJSC	  = adebug.o   err.o      fp.o       signal.o   stack.o
OBJSF	  =
MANSEC	  = 7
LIBBASE	  = libpetscsys
DIRS	  = 
LOCDIR	  = src/sys/src/error/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base




