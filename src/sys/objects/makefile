
ALL: lib 

CFLAGS    =
FFLAGS    =
CPPFLAGS  = -DPETSC_LIB_DIR='"${INSTALL_DIR}/lib/${PETSC_ARCH}"'
SOURCEC	  = gcomm.c   gtype.c   olist.c    pname.c  tagm.c \
            destroy.c gcookie.c inherit.c options.c pgname.c prefix.c init.c \
	    pinit.c ptype.c state.c aoptions.c mpinit.c subcomm.c
SOURCEF	  =
SOURCEH	  = ../../../include/petscoptions.h
OBJSC	  = gcomm.o   gtype.o   olist.o    pname.o  tagm.o \
            destroy.o gcookie.o inherit.o options.o pgname.o prefix.o init.o \
	    pinit.o ptype.o state.o aoptions.o mpinit.o subcomm.o
OBJSF	  =
MANSEC	  = Sys
LIBBASE	  = libpetsc
DIRS	  = ams
LOCDIR	  = src/sys/objects/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test

