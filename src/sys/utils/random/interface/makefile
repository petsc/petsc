
ALL: lib 

CFLAGS    = -DPETSC_LIB_DIR='"${INSTALL_DIR}/lib/${PETSC_ARCH}"' -DPETSC_ARCH='"${PETSC_ARCH}"' \
            -DPETSC_SLSUFFIX='"${SLSUFFIX}"' -DPETSC_DIR='"${PETSC_DIR}"' 
FFLAGS    =
SOURCEC	  = random.c  
SOURCEF	  =
SOURCEH	  = 
OBJSC	  = ${SOURCEC:.c=.o}
OBJSF	  =
MANSEC	  = Sys
LIBBASE	  = libpetsc
DIRS	  = 
LOCDIR	  = src/sys/utils/random/interface/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test

