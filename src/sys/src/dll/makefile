
ALL: lib 
CFLAGS    = -DPETSC_LIB_DIR='"${INSTALL_DIR}/lib/${PETSC_ARCH}"' -DPETSC_SLSUFFIX='"${SL_LINKER_SUFFIX}"'
FFLAGS    =
CPPFLAGS  = 
SOURCEC	  =  dl.c   reg.c
SOURCEF	  =
SOURCEH	  = 
OBJSC	  = dl.o   reg.o
OBJSF	  =
MANSEC	  = Sys
LIBBASE	  = libpetsc
DIRS	  = 
LOCDIR	  = src/sys/src/dll/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test


