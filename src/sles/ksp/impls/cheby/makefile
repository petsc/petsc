ALL: lib 
PETSC_DIR = ../../../..

CFLAGS   =   $(CONF) $(PETSC_INCLUDE) $(PCONF) \
           -I../../ -D__DIR__='"ksp/impls/cheby/"'
SOURCEC  = cheby.c 
SOURCEF  =
SOURCEH  = chebctx.h
OBJSC    = cheby.o 
OBJSF    =
LIBBASE  = libpetscksp
MANSEC   = 5
LOCDIR   = src/ksp/impls/cheby/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
