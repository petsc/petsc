PETSC_DIR = ../../..

ALL: lib

CFLAGS   = $(PETSC_INCLUDE) -I..  $(PCONF) \
           $(CONF) -D__DIR__='"sles/interface/"'
SOURCEC  = sles.c 
SOURCEF  = 
SOURCEH  = 
OBJSC    = sles.o 
OBJSF    =
LIBBASE  = libpetscsles
MANSEC   = 3
LOCDIR   = src/sles/interface/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
