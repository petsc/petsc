#PETSC_DIR = ../../../..
ALL: lib 

CFLAGS   = $(CPPFLAGS) $(CONF) -D__SDIR__='"$(LOCDIR)"'
FFLAGS   =
SOURCEC  = cr.c 
SOURCEF  = 
OBJSC    = cr.o 
OBJSF    =
LIBBASE  = libpetscsles
LOCDIR   = src/ksp/impls/cr/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
