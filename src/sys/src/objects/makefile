# $Id: makefile,v 1.3 1998/05/15 16:22:07 bsmith Exp bsmith $ 

ALL: lib 
CFLAGS	  = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"' -DPETSC_BOPT='"${BOPT}"' \
            -DPETSC_LDIR='"${LDIR}"' -DPETSC_DIR='"${PETSC_DIR}"' -DPETSC_SLSUFFIX='"${SLSUFFIX}"'

SOURCEC	  = cookie.c  gcomm.c   gtype.c   olist.c   pcomm.c  pname.c  tagm.c \
            destroy.c gcookie.c inherit.c options.c pgname.c prefix.c init.c pinit.c
SOURCEF	  =
SOURCEH	  = 
OBJSC	  = cookie.o  gcomm.o   gtype.o   olist.o   pcomm.o  pname.o  tagm.o \
            destroy.o gcookie.o inherit.o options.o pgname.o prefix.o init.o pinit.o
OBJSF	  =
MANSEC	  = 7
LIBBASE	  = libpetscsys
DIRS	  = 
LOCDIR	  = src/sys/src/objects/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base


