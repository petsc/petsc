# $Id: makefile,v 1.9 2000/02/02 20:08:13 bsmith Exp bsmith $ 

ALL: lib 

CFLAGS    = -DPETSC_DIR='"${PETSC_DIR}"'
FFLAGS    =
SOURCEC	  = adebug.c err.c errtrace.c errabort.c errstop.c fp.c signal.c stack.c
SOURCEF	  =
SOURCEH	  = 
OBJSC	  = adebug.o err.o errtrace.o errabort.o errstop.o fp.o signal.o stack.o
OBJSF	  =
MANSEC	  = Sys
LIBBASE	  = libpetsc
DIRS	  = 
LOCDIR	  = src/sys/src/error/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
include ${PETSC_DIR}/bmake/common_test



