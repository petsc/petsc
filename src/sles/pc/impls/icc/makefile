PETSC_DIR = ../../../..

ALL: lib

# This matrix dependency should be fixed!
CFLAGS   =  $(PETSC_INCLUDE) -I../..  \
           $(BS_INCLUDE) -I../../../../src/mat \
           $(CONF) -D__DIR__='"pc/impls/icc/"'
SOURCEC  = icc.c iccbs.c
SOURCEF  =
SOURCEH  = icc.h
OBJSC    = icc.o iccbs.o
OBJSF    =
LIBBASE  = libpetscsles
DIRS     = 
MANSEC   = 4

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
