#PETSC_DIR = ../../../../..

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__DIR__='"$(LOCDIR)"' 
SOURCEC  = beuler.c
SOURCEF  =
SOURCEH  = 
OBJSC    = beuler.o
OBJSF    =
LIBBASE  = libpetscts
MANSEC   = t
LOCDIR   = src/ts/impls/implicit/beuler/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

