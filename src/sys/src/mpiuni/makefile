# $Id: makefile,v 1.179 1997/08/15 16:24:04 bsmith Exp bsmith $ 

ALL: lib

CFLAGS	 = $(CPPFLAGS) $(CONF) -D__SDIR__='"$(LOCDIR)"'
SOURCEC	 = mpi.c
SOURCEF	 = 
SOURCEH	 = mpi.h
OBJSC	 = mpi.o
OBJSF	 =
LIBBASE	 = libmpiuni
LOCDIR   = src/mpiuni/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
