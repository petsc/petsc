# $Id: makefile,v 1.11 2000/08/24 16:06:32 balay Exp balay $ 

ALL: lib 

CFLAGS    =
FFLAGS    =
CPPFLAGS  = -DPETSC_LIB_DIR='"${PETSC_LIB_DIR}"'
SOURCEC	  = cookie.c  gcomm.c   gtype.c   olist.c   pcomm.c  pname.c  tagm.c \
            destroy.c gcookie.c inherit.c options.c pgname.c prefix.c init.c \
	    pinit.c ptype.c
SOURCEF	  =
SOURCEH	  = 
OBJSC	  = cookie.o  gcomm.o   gtype.o   olist.o   pcomm.o  pname.o  tagm.o \
            destroy.o gcookie.o inherit.o options.o pgname.o prefix.o init.o \
	    pinit.o ptype.o
OBJSF	  =
MANSEC	  = Sys
LIBBASE	  = libpetsc
DIRS	  = ams
LOCDIR	  = src/sys/src/objects/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
include ${PETSC_DIR}/bmake/common_test

