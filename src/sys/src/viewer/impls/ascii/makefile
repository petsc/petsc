PETSC_DIR = ../../../..


ALL: lib 
CFLAGS    =  -I../.. $(PETSC_INCLUDE) \
            $(CONF) -D__DIR__='"viewer/impls/file/"'
SOURCEC   = filev.c
SOURCEF   =
SOURCEH   =
OBJSC     = filev.o
OBJSF     =
MANSEC    = 7
LIBBASE   = libpetscsys
DIRS      = 

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
