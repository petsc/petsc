# $Id: makefile,v 1.32 1998/01/17 17:36:08 bsmith Exp balay $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} ${CONF} -D__SDIR__='"${LOCDIR}"'
SOURCEC  = itcl.c itfunc.c iguess.c itcreate.c iterativ.c itres.c itregis.c \
           xmon.c eige.c dlregis.c
SOURCEF  = 
SOURCEH  = 
OBJSC    = itcl.o itfunc.o iguess.o itcreate.o iterativ.o itres.o itregis.o \
           xmon.o eige.o dlregis.o
LIBBASE  = libpetscsles
DIRS     =
MANSEC   = 5
LOCDIR   = src/ksp/interface/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
