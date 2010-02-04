
ALL: lib 

CFLAGS    = 
FFLAGS    =
SOURCEC	  = arch.c fhost.c fuser.c memc.c mpiu.c psleep.c sortd.c sorti.c \
            str.c sortip.c pbarrier.c pdisplay.c ctable.c psplit.c \
	    select.c mpimesg.c sseenabled.c mpitr.c  
SOURCEF	  =
SOURCEH	  = 
OBJSC	  = ${SOURCEC:.c=.o}
OBJSF	  =
MANSEC	  = Sys
LIBBASE	  = libpetscsys
DIRS	  = ftn-kernels
LOCDIR	  = src/sys/utils/

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test

