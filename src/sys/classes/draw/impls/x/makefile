-include ../../../../../../petscdir.mk
#requirespackage 'PETSC_HAVE_X'

CPPFLAGS  = ${X_INCLUDE}
SOURCEC   = xinit.c ximage.c xcolor.c xops.c xioerr.c xtext.c xtone.c drawopenx.c
MANSEC    = Sys
SUBMANSEC = Draw
LIBBASE   = libpetscsys

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
