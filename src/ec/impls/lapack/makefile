ALL: lib

#PETSC_DIR = ../../../../

CFLAGS   = $(CPPFLAGS) $(CONF) -D__DIR__='"$(LOCDIR)"' 
SOURCEC  = gmres.c borthog.c
SOURCEH  = gmresp.h
SOURCEF  = 
OBJSC    = gmres.o borthog.o
OBJSF    = 
LIBBASE  = libpetscksp
MANSEC   = 5
LOCDIR   = src/ksp/impls/gmres/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
