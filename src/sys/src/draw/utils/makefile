# $Id: makefile,v 1.25 1997/10/10 04:04:54 bsmith Exp bsmith $

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__SDIR__='"$(LOCDIR)"' 
SOURCEC  = axis.c lg.c dscatter.c hists.c zoom.c
SOURCEF  =
SOURCEH  = 
OBJSC    = axis.o lg.o dscatter.o hists.o zoom.o
OBJSF    =
LIBBASE  = libpetscsys
MANSEC   = x
LOCDIR   = src/draw/utils/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
