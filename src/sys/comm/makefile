#requireslanguage CXXONLY
#requirespackage  'PETSC_HAVE_SIEVE'
ALL: lib

CFLAGS    =
FFLAGS    =
SOURCEC	  = sectioncopy.c
SOURCEF	  =
SOURCEH	  =
OBJSC	  = ${SOURCEC:.c=.o}
OBJSF	  =
MANSEC	  = Sys
LIBBASE	  = libpetscsys
DIRS	  =
LOCDIR	  = src/sys/comm/

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test
