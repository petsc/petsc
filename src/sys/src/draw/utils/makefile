PETSC_DIR = ../../..

ALL: lib

CFLAGS   =  $(PETSC_INCLUDE) -I../impls -I.. $(PCONF)\
           $(COPT) $(CONF) -D__DIR__='"draw/utils/"'
SOURCEC  = axis.c lg.c
SOURCEF  =
SOURCEH  = 
OBJSC    = axis.o lg.o
OBJSF    =
LIBBASE  = libpetscdraw
MANSEC   = x
LOCDIR   = src/draw/utils/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
