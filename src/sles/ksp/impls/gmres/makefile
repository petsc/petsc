ALL: lib

PETSC_DIR = ../../../../

CFLAGS   =  $(CONF) $(PETSC_INCLUDE) $(PCONF) \
           -I../../ -D__DIR__='"ksp/impls/gmres/"'
SOURCEC  = gmres.c borthog.c
SOURCEH  = gmresp.h
SOURCEF  = 
OBJSC    = gmres.o borthog.o
OBJSF    = 
LIBBASE  = libpetscksp
MANSEC   = 5
LOCDIR   = src/ksp/impls/gmres/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
