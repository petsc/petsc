ALL: lib

#PETSC_DIR = ../../../../

CFLAGS   = $(CPPFLAGS) $(CONF) -D__SDIR__='"$(LOCDIR)"' 
SOURCEC  = elapack.c
SOURCEH  = 
SOURCEF  = 
OBJSC    = elapack.o
OBJSF    = 
LIBBASE  = libpetscec
MANSEC   = 5
LOCDIR   = src/ec/impls/lapack/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
