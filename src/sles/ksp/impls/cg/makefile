PETSC_DIR = ../../../..
ALL: lib 

CFLAGS   =  $(PETSC_INCLUDE) $(CONF) $(PCONF)  \
           -I../.. -D__DIR__='"ksp/impls/cg/"'
FFLAGS   =
SOURCEC  = cg.c cgeig.c
SOURCEF  =
SOURCEH  = cgctx.h 
OBJSC    = cg.o cgeig.o 
OBJSF    =
LIBBASE  = libpetscksp
MANSEC   = 5
LOCDIR   = src/ksp/impls/cg/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
