IPETSCDIR = ../../

ALL: lib 
CFLAGS    = $(OPT) $(INCLUDEDIRS) $(CONF) -D__DIR__='"sys/"' \
            -I$(IPETSCDIR)/pinclude
SOURCEC   = tr.c arch.c  file.c  nreg.c try.c comsort.c destroy.c \
            plog.c options.c
SOURCEF   =
SOURCEH   = nreg.h
OBJSC     = tr.o arch.o file.o  nreg.o try.o comsort.o destroy.o \
            plog.o options.o
OBJSF     =
WSOURCEC  = archf.c   
WOBJS     = archf.o  
MANSEC    = 7
LINCLUDE  = $(SOURCEH)
LIBBASE   = libpetscsys
DIRS      = error examples

runexamples:

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
