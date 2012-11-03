#

#requiresdefine   'PETSC_USING_F90'

CFLAGS    =
FFLAGS    =
SOURCEC   =
SOURCEF   =
SOURCEH   = petsc.h petscsys.h petsclog.h petscvec.h petscsnes.h petscdm.h petscdmda.h petscdmmesh.h petscdraw.h petscmat.h \
	    petscksp.h petscpc.h petscviewer.h petscis.h petscao.h petscpcmg.h \
	    petscts.h petscis.h90 petscvec.h90 petscmat.h90 petscdm.h90 petscdmda.h90 petscdmadda.h90 petscdmcomposite.h90 \
            petscdmmesh.h90 petscdmredundant.h90 \
            petscdef.h  petscsysdef.h petsclogdef.h petscvecdef.h petscsnesdef.h petscdmdef.h petscdmdadef.h petscdmmeshdef.h \
            petscdrawdef.h petscmatdef.h petsckspdef.h petscpcdef.h petscviewerdef.h petscisdef.h petscaodef.h petscpcmgdef.h \
	    petsctsdef.h
LIBBASE   = libpetscvec
DIRS      =
MANSEC    = Sys
LOCDIR    = include/finclude/
NOADIC    = true

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test

