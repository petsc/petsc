PETSC_DIR = ../../../..

ALL: lib

CFLAGS   =  $(PETSC_INCLUDE)  \
           -I../..   $(CONF) -D__DIR__='"pc/impls/shell/"'
SOURCEC  = shellpc.c 
SOURCEF  =
SOURCEH  = 
OBJSC    = shellpc.o 
OBJSF    =
LIBBASE  = libpetscsles
MANSEC   = 4

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
