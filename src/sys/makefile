IPETSCDIR = ../..

ALL: lib 
CFLAGS    = $(OPT) $(INCLUDEDIRS) $(CONF) -D__DIR__='"sys/"' \
            -I$(IPETSCDIR)/pinclude
SOURCEC   =
SOURCEF   =
SOURCEH   = nreg.h
OBJSC     =
OBJSF     =
WSOURCE  = 
WOBJS     = 
MANSEC    = 7
LIBBASE   = libpetscsys
DIRS      = src examples

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
