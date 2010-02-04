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
LIBBASE	  = libpetscsys
DIRS	  = 
LOCDIR	  = src/sys/plog/utils/

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test
