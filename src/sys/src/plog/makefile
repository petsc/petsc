# $Id: makefile,v 1.11 1998/05/08 16:12:56 bsmith Exp bsmith $ 

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
LOCDIR	  = src/sys/src/plog/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
