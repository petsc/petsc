ALL: lib 

CFLAGS   = $(CPPFLAGS) $(CONF) -D__SDIR__='"$(LOCDIR)"'
SOURCEC  = tfqmr.c
SOURCEF  =
OBJSC    = tfqmr.o
OBJSF    =
LIBBASE  = libpetscsles
LOCDIR   = src/ksp/impls/tfqmr/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
