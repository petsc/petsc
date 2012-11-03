
#requirespackage 'PETSC_HAVE_X'

ALL: lib 

CFLAGS    =
FFLAGS    =
CPPFLAGS  = ${X_INCLUDE} 
SOURCEC   = xinit.c   wmap.c xcolor.c xops.c text.c tone.c drawopenx.c
SOURCEF   =
SOURCEH   = ximpl.h
MANSEC    = Draw
LIBBASE   = libpetscsys
DIRS      = 
LOCDIR    = src/sys/draw/impls/x/
NOADIC    = true

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test
