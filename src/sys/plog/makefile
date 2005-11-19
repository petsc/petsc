#requirespackage    'PETSC_USE_LOG'

ALL: lib 

CFLAGS    =
FFLAGS    =
CPPFLAGS  =
SOURCEC	  = plog.c plogmpe.c ploginfo.c classLog.c stageLog.c eventLog.c
SOURCEF	  =
SOURCEH	  = ptime.h plog.h ../../../include/petsclog.h
OBJSC	  = ${SOURCEC:.c=.o}
OBJSF	  =
MANSEC	  = Profiling
LIBBASE	  = libpetsc
DIRS	  = 
LOCDIR	  = src/sys/plog/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test
