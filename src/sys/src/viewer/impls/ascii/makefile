PETSC_DIR = ../../../..


ALL: lib 
CFLAGS    =  -I../.. $(PETSC_INCLUDE) $(PCONF)\
            $(CONF) -D__DIR__='"viewer/impls/file/"'
SOURCEC   = filev.c
SOURCEF   =
SOURCEH   =
OBJSC     = filev.o
OBJSF     =
MANSEC    = v
LIBBASE   = libpetscsys
DIRS      = 
LOCDIR    = src/viewer/impls/file/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
