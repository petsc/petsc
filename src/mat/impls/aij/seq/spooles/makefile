
#requirespackage    'PETSC_HAVE_SPOOLES'
#requiresprecision  double

ALL: lib

CFLAGS   = ${SPOOLES_INCLUDE}
FFLAGS   =
SOURCEC  = aijspooles.c spoolesOptions.c spooles.c
SOURCEF  = 
SOURCEH  = spooles.h
OBJSC    = aijspooles.o spoolesOptions.o spooles.o
OBJSF    = 
LIBBASE  = libpetscmat
DIRS     = 
MANSEC   = Mat
LOCDIR   = src/mat/impls/aij/seq/spooles/

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test
