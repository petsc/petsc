PETSC_DIR = ../../../..

ALL: lib

CFLAGS   =  $(PETSC_INCLUDE) -I../..  $(PCONF)\
          $(CONF) -I$(PETSC_DIR)/src/ksp \
          -I$(PETSC_DIR)/src/ksp/impls/qcg -D__DIR__='"snes/impls/umtr/"'
SOURCEC  = umtr.c
SOURCEF  =
SOURCEH  = umtr.h
OBJSC    = umtr.o
OBJSF    =
LIBBASE  = libpetscsnes
MANSEC   = n
LOCDIR   = src/snes/impls/umtr/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)

