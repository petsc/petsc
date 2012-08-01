
#requirespackage   'PETSC_HAVE_CLIQUE'

ALL: lib

CFLAGS   = 
FFLAGS   =
SOURCEC	 = clique.cxx
SOURCEF	 =
SOURCEH	 = matcliqueimpl.h
OBJSC	 = $(SOURCEC:.cxx=.o)
OBJSF	 =
LIBBASE	 = libpetscmat
DIRS	 =
MANSEC	 = Mat
LOCDIR	 = src/mat/impls/aij/mpi/clique/

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test
