PETSC_DIR = ../../../..


ALL: lib 
CFLAGS    =  -I../.. $(PETSC_INCLUDE) $(PCONF)\
            $(CONF) -D__DIR__='"viewer/impls/string/"'
SOURCEC   = stringv.c
SOURCEF   =
SOURCEH   =
OBJSC     = stringv.o
OBJSF     =
MANSEC    = v
LIBBASE   = libpetscsys
DIRS      = 
LOCDIR    = src/viewer/impls/string/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
