ALL: lib 
PETSC_DIR = ../../../../

CFLAGS   = $(CPPFLAGS) $(CONF) -D__DIR__='"$(LOCDIR)"'
FFLAGS   =
SOURCEC  = qcg.c
SOURCEF  =
SOURCEH  = qcg.h 
OBJSC    = qcg.o
OBJSF    =
LIBBASE  = libpetscksp
LINCLUDE = 
LOCDIR   = src/ksp/impls/qcg/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
