# $Id: makefile,v 1.22 2001/04/05 21:06:52 balay Exp buschelm $ 

ALL: lib 

CFLAGS    = -DPETSC_BOPT='"${BOPT}"' -DPETSC_LIB_DIR='"${PETSC_LIB_DIR}"' -DPETSC_ARCH='"${PETSC_ARCH}"' \
            -DPETSC_DIR='"${PETSC_DIR}"' -DPETSC_SLSUFFIX='"${SLSUFFIX}"'
FFLAGS    =
SOURCEC	  = arch.c fhost.c fuser.c memc.c mpiu.c psleep.c random.c sortd.c sorti.c \
            str.c sortip.c pbarrier.c pdisplay.c ctable.c psplit.c \
	    select.c mpimesg.c sseenabled.c
SOURCEF	  =
SOURCEH	  = dlregis.h
OBJSC	  = arch.o fhost.o fuser.o memc.o mpiu.o psleep.o random.o sortd.o sorti.o \
            str.o sortip.o pbarrier.o pdisplay.o ctable.o psplit.o \
	    select.o mpimesg.o sseenabled.o
OBJSF	  =
MANSEC	  = Sys
LIBBASE	  = libpetsc
DIRS	  = 
LOCDIR	  = src/sys/src/utils/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
include ${PETSC_DIR}/bmake/common_test

