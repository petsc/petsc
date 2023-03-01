-include ../../../../../../petscdir.mk
#requirespackage  'PETSC_HAVE_SAWS'

CFLAGS    = -DSAWS_HOME='"${SAWS_HOME}"'
SOURCEC   = ams.c amsopen.c
SOURCEF   =
SOURCEH   = ../../../../../../include/petscviewersaws.h
MANSEC    = Sys
SUBMANSEC = Viewer
LIBBASE   = libpetscsys
DIRS      =

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test




