PETSC_DIR = ../../..

ALL: lib

CFLAGS   = $(PETSC_INCLUDE) -I.. $(CONF) -D__DIR__='"draw/src/"' $(PCONF)
SOURCEC  = draw.c 
SOURCEF  =
SOURCEH  = 
OBJSC    = draw.o 
OBJSF    =
LIBBASE  = libpetscdraw
MANSEC   = x

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
