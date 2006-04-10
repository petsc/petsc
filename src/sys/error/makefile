
ALL: lib 

CFLAGS    =
FFLAGS    =
CPPFLAGS  = -DPETSC_LIB_DIR='"${INSTALL_DIR}/lib/${PETSC_ARCH}"'
SOURCEC	  = adebug.c err.c errtrace.c errabort.c errstop.c fp.c signal.c stack.c
SOURCEF	  =
SOURCEH	  = ../../../include/petscerror.h 
OBJSC	  = adebug.o err.o errtrace.o errabort.o errstop.o fp.o signal.o stack.o
OBJSF	  =
MANSEC	  = Sys
LIBBASE	  = libpetsc
DIRS	  = 
LOCDIR	  = src/sys/error/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test



