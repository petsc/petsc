# $Id: makefile,v 1.179 1997/08/15 16:24:04 bsmith Exp bsmith $

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__SDIR__='"$(LOCDIR)"' 
SOURCEC  = axis.c lg.c dscatter.c
SOURCEF  =
SOURCEH  = 
OBJSC    = axis.o lg.o dscatter.o
OBJSF    =
LIBBASE  = libpetscsys
MANSEC   = x
LOCDIR   = src/draw/utils/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
