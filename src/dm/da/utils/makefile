#PETSC_DIR = ../../..

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__SDIR__='"$(LOCDIR)"'
SOURCEC  = fdda.c
SOURCEF  = 
SOURCEH  = 
OBJSC    = fdda.o
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = 6
LOCDIR   = src/da/utils/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
