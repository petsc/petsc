# $Id: makefile,v 1.16 2000/05/18 16:49:16 balay Exp balay $ 

ALL: lib 

CFLAGS    = -DPETSC_BOPT='"${BOPT}"' -DPETSC_LDIR='"${LDIR}"' -DPETSC_ARCH='"${PETSC_ARCH}"' \
            -DPETSC_DIR='"${PETSC_DIR}"' -DPETSC_SLSUFFIX='"${SLSUFFIX}"'
FFLAGS    =
SOURCEC	  = arch.c fhost.c fuser.c memc.c mpiu.c psleep.c random.c sortd.c sorti.c \
            str.c sortip.c pbarrier.c pdisplay.c ctable.c psplit.c asciredstubs.c select.c
SOURCEF	  =
SOURCEH	  = dlregis.h
OBJSC	  = arch.o fhost.o fuser.o memc.o mpiu.o psleep.o random.o sortd.o sorti.o \
            str.o sortip.o pbarrier.o pdisplay.o ctable.o psplit.o asciredstubs.o select.o
OBJSF	  =
MANSEC	  = Sys
LIBBASE	  = libpetsc
DIRS	  = 
LOCDIR	  = src/sys/src/utils/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base



