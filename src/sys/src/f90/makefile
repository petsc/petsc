
#requirespackage 'PETSC_HAVE_F90_H'
ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = f90_common.c
SOURCEF  = 
SOURCEH  = f90_IRIX.c f90_alpha.c f90_hpux.c f90_intel8.c f90_rs6000.c f90_solaris_old.c f90_win32.c \
           f90_absoft.c f90_intel.c f90_nag.c f90_solaris.c f90_t3e.c
OBJSC    = f90_common.o
OBJSF    = 
LIBBASE  = libpetsc
MANSEC   = Sys
DIRS     = 
LOCDIR   = src/sys/src/f90/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test


