PETSC_DIR = ../../..

ALL: lib

CFLAGS   = $(PETSC_INCLUDE) -I.. $(CONF) $(PCONF) -D__DIR__='"da/src/"'
SOURCEC  = da2.c da1.c da3.c
SOURCEF  = 
SOURCEH  = 
OBJSC    = da2.o da1.o da3.o
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = 6
LOCDIR   = src/da/src/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
