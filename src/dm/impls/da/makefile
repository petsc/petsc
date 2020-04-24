-include ../petscdir.mk
ALL: lib

CFLAGS   = ${MATLAB_INCLUDE}
FFLAGS   =
SOURCEC  = da2.c da1.c da3.c daghost.c dacorn.c dagtol.c daltol.c \
           daindex.c dascatter.c dacreate.c dadestroy.c dalocal.c \
           dadist.c daview.c dasub.c gr1.c gr2.c dagtona.c \
	   dainterp.c dapf.c dagetarray.c dagetelem.c da.c dareg.c \
           fdda.c grvtk.c dageometry.c dadd.c dapreallocate.c grglvis.c
SOURCEH  = ../../../../include/petsc/private/dmdaimpl.h ../../../../include/petscdmda.h ../../../../include/petscdmdatypes.h
LIBBASE  = libpetscdm
DIRS     = usfft hypre
LOCDIR   = src/dm/impls/da/
MANSEC   = DM
SUBMANSEC= DMDA

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

