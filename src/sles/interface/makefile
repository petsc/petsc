# $Id: makefile,v 1.179 1997/08/15 16:24:04 bsmith Exp bsmith $ 

ALL: lib

CFLAGS   =  $(CPPFLAGS) $(CONF) -D__SDIR__='"$(LOCDIR)"'
SOURCEC  = sles.c 
SOURCEF  = 
SOURCEH  = 
OBJSC    = sles.o 
OBJSF    =
LIBBASE  = libpetscsles
MANSEC   = 3
LOCDIR   = src/sles/interface/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
