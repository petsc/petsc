PETSC_DIR = ../../..

ALL: lib

CFLAGS   = $(CPPFLAGS) -I.. $(CONF) -D__DIR__='"$(LOCDIR)"'
SOURCEC  = ts.c tsreg.c tsregall.c
SOURCEF  =
SOURCEH  = 
OBJSC    = ts.o tsreg.o tsregall.o
OBJSF    =
LIBBASE  = libpetscts
MANSEC   = t
LOCDIR   = src/ts/interface/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

