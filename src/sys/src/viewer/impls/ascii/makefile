IPETSCDIR = ../../../..


ALL: lib 
CFLAGS    = $(OPT) -I../.. $(INCLUDEDIRS) -I$(IPETSCDIR)/pinclude \
            $(XINCLUDEDIR) $(CONF) -D__DIR__='"viewer/impls/file/"'
SOURCEC   = filev.c
SOURCEF   =
SOURCEH   =
OBJSC     = filev.o
OBJSF     =
WSOURCE  =   
WOBJS     = 
MANSEC    = x
LIBBASE   = libpetscsys
DIRS      = 

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
