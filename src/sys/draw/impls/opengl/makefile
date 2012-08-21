
#requirespackage 'PETSC_HAVE_OPENGL'

ALL: lib 

CFLAGS    =
FFLAGS    =
CPPFLAGS  = ${OPENGL_INCLUDE} 
SOURCEC   = openglops.c
SOURCEF   =
SOURCEH   = 
OBJSC     = openglops.o
OBJSF     =
MANSEC    = Draw
LIBBASE   = libpetscsys
DIRS      = 
LOCDIR    = src/sys/draw/impls/opengl/
NOADIC    = true

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test
