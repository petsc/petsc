PETSC_DIR = ../../../..


ALL: lib 
CFLAGS    = $(CPPFLAGS) $(CONF) -D__DIR__='"$(LOCDIR)"' 
SOURCEC   = filev.c
SOURCEF   =
SOURCEH   =
OBJSC     = filev.o
OBJSF     =
MANSEC    = v
LIBBASE   = libpetscsys
DIRS      = 
LOCDIR    = src/viewer/impls/file/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
