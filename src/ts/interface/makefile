PETSC_DIR = ../../..

ALL: lib

CFLAGS   = $(PETSC_INCLUDE) -I..  $(PCONF) \
           $(CONF) -D__DIR__='"ts/interface/"'
SOURCEC  = ts.c tsreg.c tsregall.c
SOURCEF  =
SOURCEH  = 
OBJSC    = ts.o tsreg.o tsregall.o
OBJSF    =
LIBBASE  = libpetscts
MANSEC   = t
LOCDIR   = src/ts/interface/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)

