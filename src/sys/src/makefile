IPETSCDIR = ../../..

ALL: lib 
CFLAGS    =  $(PETSC_INCLUDE) $(CONF)\
            -D__DIR__='"sys/src/"'
SOURCEC   = adebug.c err.c signal.c fp.c tr.c arch.c file.c nreg.c \
            mpiu.c comsort.c destroy.c plog.c options.c mal.c \
            sysio.c
SOURCEF   =
SOURCEH   =
OBJSC     = adebug.o err.o signal.o fp.o tr.o arch.o file.o nreg.o \
            mpiu.o comsort.o destroy.o  plog.o options.o mal.o \
            sysio.o
OBJSF     =
MANSEC    = 7
LIBBASE   = libpetscsys
DIRS      = 

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
