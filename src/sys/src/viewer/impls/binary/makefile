PETSC_DIR = ../../../..

ALL: lib 
CFLAGS    =  -I../.. $(PETSC_INCLUDE) $(PCONF)\
            $(XINCLUDEDIR) $(CONF) -D__DIR__='"viewer/impls/file/"'
SOURCEC   = binv.c
SOURCEF   =
SOURCEH   =
OBJSC     = binv.o
OBJSF     =
MANSEC    = 7
LIBBASE   = libpetscsys
DIRS      = 

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
