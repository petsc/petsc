PETSC_DIR = ../../../..

ALL: lib

CFLAGS   =  $(PETSC_INCLUDE) -I../..  \
          $(CONF) -D__DIR__='"snes/impls/test/"'
SOURCEC  = snestest.c
SOURCEF  =
SOURCEH  = 
OBJSC    = snestest.o
OBJSF    =
LIBBASE  = libpetscsnes
MANSEC   = n

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)

