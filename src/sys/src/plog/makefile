# $Id: makefile,v 1.10 1998/04/23 23:28:27 balay Exp bsmith $ 

ALL: lib 
CFLAGS	  = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"'
SOURCEC	  = plog.c plogmpe.c ploginfo.c mpitr.c
SOURCEF	  =
SOURCEH	  = 
OBJSC	  = plog.o plogmpe.o ploginfo.o mpitr.o
OBJSF	  =
MANSEC	  = p
LIBBASE	  = libpetscsys
DIRS	  = 
LOCDIR	  = src/plog/src/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
