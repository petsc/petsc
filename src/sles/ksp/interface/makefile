# $Id: makefile,v 1.42 2000/08/17 04:52:21 bsmith Exp balay $ 

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

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test
