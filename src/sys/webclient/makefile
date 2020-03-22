#requirespackage  'PETSC_HAVE_SSL'
#requiresdefine   'PETSC_USE_SOCKET_VIEWER'

ALL: lib

CFLAGS    =
FFLAGS    =
CPPFLAGS  =
SOURCEC	  = client.c google.c box.c textbelt.c globus.c tellmycell.c
SOURCEF	  =
SOURCEH	  = ../../../include/petscwebclient.h
MANSEC	  = Sys
LIBBASE	  = libpetscsys
DIRS	  = tutorials
LOCDIR	  = src/sys/webclient/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test



