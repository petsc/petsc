PETSC_DIR = ../../../..
ALL: lib 

CFLAGS   =  $(CPPFLAGS) $(CONF) -D__DIR__='"$(LOCDIR)"' -I../../
SOURCEC  = bcgs.c
SOURCEF  =
OBJSC    = bcgs.o
OBJSF    =
LIBBASE  = libpetscksp
LOCDIR   = src/ksp/impls/bcgs/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
