# $Id: makefile,v 1.24 1997/08/22 15:16:30 bsmith Exp bsmith $

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__SDIR__='"$(LOCDIR)"' 
SOURCEC  = axis.c lg.c dscatter.c hists.c
SOURCEF  =
SOURCEH  = 
OBJSC    = axis.o lg.o dscatter.o hists.o
OBJSF    =
LIBBASE  = libpetscsys
MANSEC   = x
LOCDIR   = src/draw/utils/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
