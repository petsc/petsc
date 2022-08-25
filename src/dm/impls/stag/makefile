-include ../../../../petscdir.mk

SOURCEC  = stag.c stag1d.c stag2d.c stag3d.c stagda.c stagintern.c stagmulti.c stagstencil.c stagutils.c
SOURCEF  =
SOURCEH  = ../../../../include/petscdmstag.h ../../../../include/petsc/private/dmstagimpl.h
DIRS     = tests tutorials
LIBBASE  = libpetscdm
MANSEC   = DM
SUBMANSEC= DMStag
LOCDIR   = src/dm/impls/stag/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
