# $Id: makefile,v 1.6 1998/06/15 15:42:20 balay Exp bsmith $ 

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
MANSEC	  = 7
LIBBASE	  = libpetsc
DIRS	  = 
LOCDIR	  = src/sys/src/objects/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base


