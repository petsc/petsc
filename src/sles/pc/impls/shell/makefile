PETSC_DIR = ../../../..

ALL: lib

CFLAGS   =  $(PETSC_INCLUDE) $(PCONF) \
           -I../..   $(CONF) -D__DIR__='"pc/impls/shell/"'
SOURCEC  = shellpc.c 
SOURCEF  =
SOURCEH  = 
OBJSC    = shellpc.o 
OBJSF    =
LIBBASE  = libpetscsles
MANSEC   = 4
LOCDIR   = src/pc/impls/shell/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
