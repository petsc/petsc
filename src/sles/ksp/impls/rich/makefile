ALL: lib 
PETSC_DIR = ../../../..

CFLAGS   =  $(CONF) $(PETSC_INCLUDE) -I../.. $(PCONF) \
           -D__DIR__='"ksp/impls/rich/"'
SOURCEC  = rich.c
SOURCEF  =
SOURCEH  = richctx.h
OBJSC    = rich.o 
OBJSF    =
LIBBASE  = libpetscksp
MANSEC   = 5
LOCDIR   = src/ksp/impls/rich/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
