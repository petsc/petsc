
#requirespackage  'PETSC_HAVE_AMS'

ALL: lib 

CFLAGS   = -DAMS_HOME='"${AMS_HOME}"'
FFLAGS   =
SOURCEC  = ams.c amsopen.c
SOURCEF  =
SOURCEH  =
OBJSC    = ams.o amsopen.o
OBJSF    =
MANSEC   = Viewer
LIBBASE  = libpetsc
DIRS     = 
LOCDIR   = src/sys/src/viewer/impls/ams/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test



