PETSC_DIR = ../../../..

ALL: lib

# This matrix dependency should be fixed!
CFLAGS   =  $(PETSC_INCLUDE) -I../..  $(PCONF) \
           $(BS_INCLUDE) -I../../../../src/mat \
           $(CONF) -D__DIR__='"pc/impls/icc/"'
SOURCEC  = icc.c
SOURCEF  =
SOURCEH  = icc.h
OBJSC    = icc.o
OBJSF    =
LIBBASE  = libpetscsles
DIRS     = 
MANSEC   = 4
LOCDIR   = src/pc/impls/icc/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
