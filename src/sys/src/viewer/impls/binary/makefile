IPETSCDIR = ../../../..

ALL: lib 
CFLAGS    = $(OPT) -I../.. $(PETSC_INCLUDE) -I$(IPETSCDIR)/pinclude \
            $(XINCLUDEDIR) $(CONF) -D__DIR__='"viewer/impls/file/"'
SOURCEC   = binv.c
SOURCEF   =
SOURCEH   =
OBJSC     = binv.o
OBJSF     =
MANSEC    = 7
LIBBASE   = libpetscsys
DIRS      = 

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
