# $Id: makefile,v 1.179 1997/08/15 16:24:04 bsmith Exp bsmith $ 

ALL: lib

CFLAGS   =  $(CPPFLAGS) $(CONF) -D__SDIR__='"$(LOCDIR)"'
SOURCEC  = precon.c pcset.c pcregis.c pcnull.c
SOURCEF  =
SOURCEH  = 
OBJSC    = precon.o pcset.o pcregis.o pcnull.o
OBJSF    =
LIBBASE  = libpetscsles
MANSEC   = 4
LOCDIR   = src/pc/interface/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
