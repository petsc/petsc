PETSC_DIR = ../../..

ALL: lib

CFLAGS   = $(PETSC_INCLUDE) -I.. $(CONF) $(PCONF) -D__DIR__='"da/src/"'
SOURCEC  = da2.c da1.c da3.c daghost.c dacorn.c dagtol.c daltol.c \
           daltog.c daindex.c dascatter.c dadestroy.c dalocal.c \
           dadist.c daview.c
SOURCEF  = 
SOURCEH  = 
OBJSC    = da2.o da1.o da3.o daghost.o dacorn.o dagtol.o daltol.o \
           daltog.o daindex.o dascatter.o dadestroy.o dalocal.o \
           dadist.o daview.o
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = 6
LOCDIR   = src/da/src/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
