# $Id: makefile,v 1.2 1998/05/08 16:12:54 bsmith Exp bsmith $ 

ALL: lib 
CFLAGS	  = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"' -DPETSC_BOPT='"${BOPT}"' \
            -DPETSC_LDIR='"${LDIR}"' -DPETSC_DIR='"${PETSC_DIR}"' -DPETSC_SLSUFFIX='"${SLSUFFIX}"'

SOURCEC	  = arch.c fhost.c fuser.c memc.c mpiu.c nreg.c psleep.c random.c sortd.c sorti.c \
            str.c sortip.c pbarrier.c pdisplay.c
SOURCEF	  =
SOURCEH	  = 
OBJSC	  = arch.o fhost.o fuser.o memc.o mpiu.o nreg.o psleep.o random.o sortd.o sorti.o \
            str.o sortip.o pbarrier.o pdisplay.o
OBJSF	  =
MANSEC	  = 7
LIBBASE	  = libpetscsys
DIRS	  = 
LOCDIR	  = src/sys/src/utils

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base



