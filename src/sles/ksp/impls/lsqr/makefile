ALL: lib 

CFLAGS   = $(CPPFLAGS) $(CONF) -D__SDIR__='"$(LOCDIR)"' 
SOURCEC  = lsqr.c
SOURCEF  =
OBJSC    = lsqr.o
OBJSF    =
LIBBASE  = libpetscsles
LOCDIR   = src/ksp/impls/lsqr/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
