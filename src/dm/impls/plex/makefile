-include ../petscdir.mk
ALL: lib

CPPFLAGS = ${NETCFD_INCLUDE} ${EXODUSII_INCLUDE}
CFLAGS   =
FFLAGS   =
SOURCEC  = plexcreate.c plex.c plexpartition.c plexdistribute.c plexrefine.c plexadapt.c plexcoarsen.c plexinterpolate.c plexpreallocate.c plexreorder.c plexgeometry.c plexsubmesh.c plexhdf5.c plexhdf5xdmf.c plexexodusii.c plexgmsh.c plexfluent.c plexcgns.c plexmed.c plexply.c plexvtk.c plexpoint.c plexvtu.c plexfem.c plexfvm.c plexindices.c plextree.c plexgenerate.c plexorient.c plexnatural.c plexproject.c plexglvis.c glexg.c petscpartmatpart.c plexcheckinterface.c plexsection.c plexhpddm.c plexegads.c
SOURCEF  =
SOURCEH  =
DIRS     = generators tests tutorials
LIBBASE  = libpetscdm
MANSEC   = DM
SUBMANSEC= DMPLEX
LOCDIR   = src/dm/impls/plex/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
