ALL: lib 

CFLAGS   = $(CPPFLAGS) $(CONF) -D__SDIR__='"$(LOCDIR)"' 
SOURCEC  = cheby.c 
SOURCEF  =
SOURCEH  = chebctx.h
OBJSC    = cheby.o 
OBJSF    =
LIBBASE  = libpetscsles
MANSEC   = 5
LOCDIR   = src/ksp/impls/cheby/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
