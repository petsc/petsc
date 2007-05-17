#requirespackage    'PETSC_USE_LOG'

ALL: lib 

CFLAGS    =
FFLAGS    =
CPPFLAGS  =
SOURCEC	  = classLog.c stageLog.c eventLog.c
SOURCEF	  =
SOURCEH	  = 
OBJSC	  = ${SOURCEC:.c=.o}
OBJSF	  =
MANSEC	  = Profiling
LIBBASE	  = libpetsc
DIRS	  = 
LOCDIR	  = src/sys/plog/utils/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test
