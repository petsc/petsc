PETSC_DIR = ../../../..

ALL: lib

CFLAGS   =  $(PETSC_INCLUDE) -I../..  $(PCONF)\
          $(CONF) -D__DIR__='"snes/impls/tr/"'
SOURCEC  = tr.c
SOURCEF  =
SOURCEH  = tr.h
OBJSC    = tr.o
OBJSF    =
LIBBASE  = libpetscsnes
MANSEC   = n

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)

