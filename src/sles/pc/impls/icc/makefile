PETSC_DIR = ../../../..

ALL: lib

# This matrix dependency should be fixed!
CFLAGS   = $(CPPFLAGS) $(CONF) -D__DIR__='"$(LOCDIR)"'   \
           $(BS_INCLUDE) -I../../../../src/mat -I../..
SOURCEC  = icc.c
SOURCEF  =
SOURCEH  = icc.h
OBJSC    = icc.o
OBJSF    =
LIBBASE  = libpetscsles
DIRS     = 
MANSEC   = 4
LOCDIR   = src/pc/impls/icc/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
