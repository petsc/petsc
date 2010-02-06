
#requirespackage   'PETSC_HAVE_SILO'

ALL: lib 

CFLAGS    =
FFLAGS    =
SOURCEC   = vsilo.c
SOURCEH   = vsilo.h
OBJSC     = ${SOURCEC:.c=.o}
MANSEC    = Viewer
LIBBASE   = libpetscsys
DIRS      = 
LOCDIR    = src/sys/viewer/impls/silo/

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test
