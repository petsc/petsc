PETSC_DIR = ../../../..

ALL: lib

CFLAGS   =  $(PETSC_INCLUDE) -I../..  $(PCONF)\
          $(CONF) -D__DIR__='"ts/impls/explicit/"'
SOURCEC  = 
SOURCEF  =
SOURCEH  = 
OBJSC    = 
OBJSF    =
LIBBASE  = libpetscts
MANSEC   = t
LOCDIR   = src/ts/impls/explicit/
DIRS     = euler

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

