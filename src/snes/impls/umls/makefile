#PETSC_DIR = ../../../..

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__DIR__='"$(LOCDIR)"'
SOURCEC  = umls.c search.c
SOURCEF  = 
SOURCEH  = umls.h
OBJSC    = umls.o search.o
OBJSF    = 
LIBBASE  = libpetscsnes
MANSEC   = n
LOCDIR   = src/snes/impls/umls/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

