PETSC_DIR = ../../..

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__DIR__='"$(LOCDIR)"' -I../impls -I.. 
SOURCEC  = axis.c lg.c
SOURCEF  =
SOURCEH  = 
OBJSC    = axis.o lg.o
OBJSF    =
LIBBASE  = libpetscsys
MANSEC   = x
LOCDIR   = src/draw/utils/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
