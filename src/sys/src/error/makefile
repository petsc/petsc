# $Id: makefile,v 1.11 2001/08/22 17:59:17 balay Exp $ 

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



