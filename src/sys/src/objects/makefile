# $Id: makefile,v 1.7 1999/01/12 23:14:21 bsmith Exp balay $ 

ALL: lib 

CFLAGS    =
FFLAGS    =
CPPFLAGS  = -DPETSC_LDIR='"${LDIR}"'
SOURCEC	  = cookie.c  gcomm.c   gtype.c   olist.c   pcomm.c  pname.c  tagm.c \
            destroy.c gcookie.c inherit.c options.c pgname.c prefix.c init.c pinit.c
SOURCEF	  =
SOURCEH	  = 
OBJSC	  = cookie.o  gcomm.o   gtype.o   olist.o   pcomm.o  pname.o  tagm.o \
            destroy.o gcookie.o inherit.o options.o pgname.o prefix.o init.o pinit.o
OBJSF	  =
MANSEC	  = Sys
LIBBASE	  = libpetsc
DIRS	  = 
LOCDIR	  = src/sys/src/objects/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base


