PETSC_DIR = ../../../..


ALL: lib 
CFLAGS    = $(CPPFLAGS) $(CONF) -D__DIR__='"$(LOCDIR)"' -I../..
SOURCEC   = stringv.c
SOURCEF   =
SOURCEH   =
OBJSC     = stringv.o
OBJSF     =
MANSEC    = v
LIBBASE   = libpetscsys
DIRS      = 
LOCDIR    = src/viewer/impls/string/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
