PETSC_DIR = ../../..

ALL: lib

CFLAGS   =  $(PETSC_INCLUDE) -I../impls -I..\
           $(COPT) $(CONF) -D__DIR__='"draw/utils/"'
SOURCEC  = axis.c lg.c
SOURCEF  =
SOURCEH  = 
OBJSC    = axis.o lg.o
OBJSF    =
LIBBASE  = libpetscdraw
MANSEC   = x

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
