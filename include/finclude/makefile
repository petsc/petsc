#

#requirespackage   'PETSC_USING_F90'

CFLAGS    =
FFLAGS    =
SOURCEC   =
SOURCEF   =
SOURCEH   = petsc.h petscsys.h petsclog.h petscvec.h petscsnes.h petscda.h petscdraw.h petscmat.h \
	    petscksp.h petscpc.h petscviewer.h petscis.h petscao.h petscmg.h \
	    petscts.h petscis.h90 petscvec.h90 petscmat.h90 petscda.h90 \
            petscdef.h  petscsysdef.h petsclogdef.h petscvecdef.h petscsnesdef.h petscdadef.h petscdrawdef.h petscmatdef.h \
	    petsckspdef.h petscpcdef.h petscviewerdef.h petscisdef.h petscaodef.h petscmgdef.h \
	    petsctsdef.h petscmeshdef.h 
OBJSC     =
OBJSF     =
LIBBASE   = libpetscvec
DIRS      = 
MANSEC    = Sys
LOCDIR    = include/finclude/
NOADIC    = true

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test

