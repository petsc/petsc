PETSC_DIR = ../../../..

ALL: lib

CFLAGS   =  $(PETSC_INCLUDE) -I../..  $(PCONF)\
           $(CONF) -D__DIR__='"snes/impls/ls/"'
SOURCEC  = ls.c
SOURCEF  =
SOURCEH  = ls.h
OBJSC    = ls.o
OBJSF    =
LIBBASE  = libpetscsnes
MANSEC   = n

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)

