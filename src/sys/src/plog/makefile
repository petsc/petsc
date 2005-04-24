
ALL: lib 

CFLAGS    =
FFLAGS    =
CPPFLAGS  = -DCONFIGURE_RUN_TIME='"${CONFIGURE_RUN_TIME}"'
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
