PETSC_DIR = ../../../..
ALL: lib 

CFLAGS   = $(CPPFLAGS) $(CONF) -D__DIR__='"$(LOCDIR)"' 
FFLAGS   =
SOURCEC  = cg.c cgeig.c
SOURCEF  =
SOURCEH  = cgctx.h 
OBJSC    = cg.o cgeig.o 
OBJSF    =
LIBBASE  = libpetscksp
MANSEC   = 5
LOCDIR   = src/ksp/impls/cg/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
