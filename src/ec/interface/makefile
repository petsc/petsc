#PETSC_DIR = ../../..

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__SDIR__='"$(LOCDIR)"'
SOURCEC  = ec.c ecall.c
SOURCEF  = 
SOURCEH  = 
OBJSC    = ec.o ecall.o
LIBBASE  = libpetscec
DIRS     =
MANSEC   = e
LOCDIR   = src/ec/interface/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
