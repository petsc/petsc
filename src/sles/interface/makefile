PETSC_DIR = ../../..

ALL: lib

CFLAGS   =  $(CPPFLAGS) $(CONF) -D__DIR__='"$(LOCDIR)"'
SOURCEC  = sles.c 
SOURCEF  = 
SOURCEH  = 
OBJSC    = sles.o 
OBJSF    =
LIBBASE  = libpetscsles
MANSEC   = 3
LOCDIR   = src/sles/interface/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
