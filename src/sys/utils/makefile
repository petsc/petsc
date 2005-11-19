
ALL: lib 

CFLAGS    = -DPETSC_LIB_DIR='"${INSTALL_DIR}/lib/${PETSC_ARCH}"' -DPETSC_ARCH='"${PETSC_ARCH}"' \
            -DPETSC_SLSUFFIX='"${SLSUFFIX}"' -DPETSC_DIR='"${PETSC_DIR}"' 
FFLAGS    =
SOURCEC	  = arch.c fhost.c fuser.c memc.c mpiu.c psleep.c random.c sortd.c sorti.c \
            str.c sortip.c pbarrier.c pdisplay.c ctable.c psplit.c \
	    select.c mpimesg.c sseenabled.c mpitr.c  
SOURCEF	  =
SOURCEH	  = 
OBJSC	  = ${SOURCEC:.c=.o}
OBJSF	  =
MANSEC	  = Sys
LIBBASE	  = libpetsc
DIRS	  = 
LOCDIR	  = src/sys/utils/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test

