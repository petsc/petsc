# $Id: makefile,v 1.41 1999/02/02 21:29:12 balay Exp bsmith $ 

ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = itcl.c itfunc.c iguess.c itcreate.c iterativ.c itres.c itregis.c \
           xmon.c eige.c dlregis.c
SOURCEF  = 
SOURCEH  = 
OBJSC    = itcl.o itfunc.o iguess.o itcreate.o iterativ.o itres.o itregis.o \
           xmon.o eige.o dlregis.o
LIBBASE  = libpetscsles
DIRS     =
MANSEC   = KSP
LOCDIR   = src/sles/ksp/interface/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
include ${PETSC_DIR}/bmake/common_test
