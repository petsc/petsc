
ALL: lib 

CFLAGS    = -DPETSC_LIB_DIR='"${INSTALL_DIR}/lib/${PETSC_ARCH}"' -DPETSC_ARCH='"${PETSC_ARCH}"' \
            -DPETSC_SLSUFFIX='"${SLSUFFIX}"' -DPETSC_DIR='"${PETSC_DIR}"' 
FFLAGS    =
SOURCEC	  = arch.c fhost.c fuser.c memc.c mpiu.c psleep.c random.c sortd.c sorti.c \
            str.c sortip.c pbarrier.c pdisplay.c ctable.c psplit.c \
	    select.c mpimesg.c sseenabled.c  
SOURCEF	  =
SOURCEH	  = 
OBJSC	  = arch.o fhost.o fuser.o memc.o mpiu.o psleep.o random.o sortd.o sorti.o \
            str.o sortip.o pbarrier.o pdisplay.o ctable.o psplit.o \
	    select.o mpimesg.o sseenabled.o 
OBJSF	  =
MANSEC	  = Sys
LIBBASE	  = libpetsc
DIRS	  = 
LOCDIR	  = src/sys/utils/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test

