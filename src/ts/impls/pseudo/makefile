PETSC_DIR = ../../../../..

ALL: lib

CFLAGS   =  $(PETSC_INCLUDE) -I../../..  $(PCONF)\
          $(CONF) -D__DIR__='"ts/impls/pseudo/posindep/"'
SOURCEC  = posindep.c
SOURCEF  =
SOURCEH  = 
OBJSC    = posindep.o
OBJSF    =
LIBBASE  = libpetscts
MANSEC   = t
LOCDIR   = src/ts/impls/pseudo/posindep/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

