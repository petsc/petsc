PETSC_DIR = ../../../..

ALL: lib

CFLAGS   =  $(PETSC_INCLUDE) -I../..  $(PCONF)\
          $(CONF) -D__DIR__='"ts/impls/exnonlin/"'
SOURCEC  = exnonlin.c
SOURCEF  =
SOURCEH  = 
OBJSC    = exnonlin.o
OBJSF    =
LIBBASE  = libpetscts
MANSEC   = t
LOCDIR   = src/ts/impls/exnonlin/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)

