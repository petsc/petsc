ALL: lib 

#PETSC_DIR = ../../../..

CFLAGS   = $(CPPFLAGS) $(CONF) -D__SDIR__='"$(LOCDIR)"'
SOURCEC  = preonly.c
SOURCEF  =
OBJSC    = preonly.o
OBJSF    =
LIBBASE  = libpetscsles
LOCDIR   = src/ksp/impls/preonly/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
