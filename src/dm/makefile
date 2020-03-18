
ALL: lib

SOURCEH  = ../../include/petscdm.h ../../include/petscdmtypes.h \
        ../../include/petsc/private/dmimpl.h ../../include/petsc/private/dmmbimpl.h \
        ../../include/petsc/private/dmnetworkimpl.h ../../include/petsc/private/dmpatchimpl.h \
        ../../include/petsc/private/dmpleximpl.h ../../include/petsc/private/dtimpl.h \
        ../../include/petsc/private/petscdsimpl.h ../../include/petsc/private/petscfeimpl.h \
        ../../include/petscdmcomposite.h ../../include/petscdmmoab.h ../../include/petscdmnetwork.h \
        ../../include/petscdmpatch.h ../../include/petscdmplex.h ../../include/petscdmlabel.h \
        ../../include/petsc/private/dmlabelimpl.h
DIRS     = impls f90-mod interface tests tutorials dt label field
LOCDIR   = src/dm/
MANSEC   = DM

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
