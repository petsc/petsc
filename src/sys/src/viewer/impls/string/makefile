# $Id: makefile,v 1.23 1997/08/22 15:15:29 bsmith Exp $ 

ALL: lib 
CFLAGS    = $(CPPFLAGS) $(CONF) -D__SDIR__='"$(LOCDIR)"' 
SOURCEC   = stringv.c
SOURCEF   =
SOURCEH   =
OBJSC     = stringv.o
OBJSF     =
MANSEC    = v
LIBBASE   = libpetscsys
DIRS      = 
LOCDIR    = src/viewer/impls/string/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
