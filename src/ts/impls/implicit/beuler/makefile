PETSC_DIR = ../../../../..

ALL: lib

CFLAGS   =  $(PETSC_INCLUDE) -I../../..  $(PCONF)\
          $(CONF) -D__DIR__='"ts/impls/implicit/beuler/"'
SOURCEC  = beuler.c
SOURCEF  =
SOURCEH  = 
OBJSC    = beuler.o
OBJSF    =
LIBBASE  = libpetscts
MANSEC   = t
LOCDIR   = src/ts/impls/implicit/beuler/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)

