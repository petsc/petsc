PETSC_DIR = ../../../..

ALL: lib

CFLAGS   =  $(CPPFLAGS) $(CONF) -D__DIR__='"$(LOCDIR)"' -I../..  
SOURCEC  = 
SOURCEF  =
SOURCEH  = 
OBJSC    = 
OBJSF    =
LIBBASE  = libpetscts
MANSEC   = t
LOCDIR   = src/ts/impls/implicit/
DIRS     = beuler

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

