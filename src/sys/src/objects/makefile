# $Id: makefile,v 1.13 2001/08/22 17:59:28 balay Exp $ 

ALL: lib 

CFLAGS    =
FFLAGS    =
CPPFLAGS  = -DPETSC_LIB_DIR='"${PETSC_LIB_DIR}"'
SOURCEC	  = cookie.c  gcomm.c   gtype.c   olist.c   pcomm.c  pname.c  tagm.c \
            destroy.c gcookie.c inherit.c options.c pgname.c prefix.c init.c \
	    pinit.c ptype.c state.c
SOURCEF	  =
SOURCEH	  = 
OBJSC	  = cookie.o  gcomm.o   gtype.o   olist.o   pcomm.o  pname.o  tagm.o \
            destroy.o gcookie.o inherit.o options.o pgname.o prefix.o init.o \
	    pinit.o ptype.o state.o
OBJSF	  =
MANSEC	  = Sys
LIBBASE	  = libpetsc
DIRS	  = ams
LOCDIR	  = src/sys/src/objects/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test

