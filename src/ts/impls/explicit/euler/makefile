PETSC_DIR = ../../../../..

ALL: lib

CFLAGS   =  $(PETSC_INCLUDE) -I../../..  $(PCONF)\
          $(CONF) -D__DIR__='"ts/impls/explicit/euler/"'
SOURCEC  = euler.c
SOURCEF  =
SOURCEH  = 
OBJSC    = euler.o
OBJSF    =
LIBBASE  = libpetscts
MANSEC   = t
LOCDIR   = src/ts/impls/explicit/euler/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

