#PETSC_DIR = ../../../..
ALL: lib 

CFLAGS   = $(CPPFLAGS) $(CONF) -D__SDIR__='"$(LOCDIR)"' 
FFLAGS   =
SOURCEC  = cg.c cgeig.c cgtype.c
SOURCEF  =
SOURCEH  = cgctx.h 
OBJSC    = cg.o cgeig.o cgtype.o
OBJSF    =
LIBBASE  = libpetscksp
MANSEC   = 5
LOCDIR   = src/ksp/impls/cg/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
