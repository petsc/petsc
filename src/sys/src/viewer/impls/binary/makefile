PETSC_DIR = ../../../..

ALL: lib 
CFLAGS    = $(CPPFLAGS) $(CONF) -D__DIR__='"$(LOCDIR)"' \
            -I../..  $(XINCLUDEDIR)
SOURCEC   = binv.c
SOURCEF   =
SOURCEH   =
OBJSC     = binv.o
OBJSF     =
MANSEC    = v
LIBBASE   = libpetscsys
DIRS      = 
LOCDIR    = src/viewer/impls/binary/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
