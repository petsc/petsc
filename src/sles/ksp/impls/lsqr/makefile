ALL: lib 

#PETSC_DIR = ../../../..

CFLAGS   = $(CPPFLAGS) $(CONF) -D__DIR__='"$(LOCDIR)"' 
SOURCEC  = lsqr.c
SOURCEF  =
OBJSC    = lsqr.o
OBJSF    =
LIBBASE  = libpetscksp
LOCDIR   = src/ksp/impls/lsqr/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
