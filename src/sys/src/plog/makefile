# $Id: makefile,v 1.8 1997/11/28 16:22:24 bsmith Exp balay $ 

ALL: lib 
CFLAGS	  = ${CPPFLAGS} ${CONF} -D__SDIR__='"${LOCDIR}"'
SOURCEC	  = plog.c plogmpe.c ploginfo.c
SOURCEF	  =
SOURCEH	  = 
OBJSC	  = plog.o plogmpe.o ploginfo.o
OBJSF	  =
MANSEC	  = p
LIBBASE	  = libpetscsys
DIRS	  = 
LOCDIR	  = src/plog/src/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
