ALL: lib 

CFLAGS   = $(CPPFLAGS) $(CONF) -D__SDIR__='"$(LOCDIR)"'
SOURCEC  = tcqmr.c
SOURCEF  =
SOURCEH  = tcqmrp.h
OBJSC    = tcqmr.o
OBJSF    = 
LIBBASE  = libpetscsles
LOCDIR   = src/ksp/impls/tcqmr/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
