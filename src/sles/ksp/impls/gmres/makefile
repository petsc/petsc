ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__SDIR__='"$(LOCDIR)"' 
SOURCEC  = gmres.c borthog.c borthog2.c borthog3.c gmres2.c gmreig.c \
           gmpre.c gmcnv.c
SOURCEH  = gmresp.h
SOURCEF  = 
OBJSC    = gmres.o borthog.o borthog2.o borthog3.o gmres2.o gmreig.o \
           gmpre.o gmcnv.o
OBJSF    = 
LIBBASE  = libpetscsles
MANSEC   = 5
LOCDIR   = src/ksp/impls/gmres/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
