# $Id: makefile,v 1.21 2001/08/22 17:59:37 balay Exp $ 

ALL: lib 

CFLAGS    =
FFLAGS    =
SOURCEC	  = plog.c plogmpe.c ploginfo.c mpitr.c classLog.c stageLog.c eventLog.c
SOURCEF	  =
SOURCEH	  = ptime.h plog.h ../../../../include/petsclog.h
OBJSC	  = ${SOURCEC:.c=.o}
OBJSF	  =
MANSEC	  = Profiling
LIBBASE	  = libpetsc
DIRS	  = 
LOCDIR	  = src/sys/src/plog/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test
