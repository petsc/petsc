IPETSCDIR = ../../../

ALL: lib 
CFLAGS    = $(OPT) $(INCLUDEDIRS) -I$(IPETSCDIR)/pinclude $(CONF)\
            -D__DIR__='"sys/error/"'
SOURCEC   = adebug.c  error.c signal.c fp.c  tr.c arch.c  file.c  nreg.c 
            try.c comsort.c destroy.c plog.c options.c mal.c
SOURCEF   =
SOURCEH   =
OBJSC     = adebug.o  error.o signal.o fp.o  tr.o arch.o file.o  nreg.o 
            try.o comsort.o destroy.o  plog.o options.o mal.o
OBJSF     =
WSOURCEC  = errorf.c  
WOBJS     = errorf.o 
MANSEC    = 7

LINCLUDE  = $(SOURCEH)
LIBBASE   = libpetscsys
DIRS      = 

runexamples:

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
