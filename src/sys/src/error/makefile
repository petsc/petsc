# $Id: makefile,v 1.10 2000/08/17 04:50:40 bsmith Exp balay $ 

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

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test



