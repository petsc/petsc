
ALL: lib

CFLAGS   = ${MATLAB_INCLUDE} ${PNETCDF_INCLUDE} 
FFLAGS   =
SOURCEC  = da2.c da1.c da3.c daghost.c dacorn.c dagtol.c daltol.c \
           daindex.c dascatter.c dacreate.c dadestroy.c dalocal.c \
           dadist.c daview.c dasub.c gr1.c gr2.c dagtona.c \
	   dainterp.c dapf.c dagetarray.c dagetelem.c da.c dareg.c fdda.c grvtk.c
OBJSC    = $(SOURCEC:.c=.o)
SOURCEH  = ../../../../include/petsc-private/daimpl.h ../../../../include/petscdmda.h 
LIBBASE  = libpetscdm
DIRS     = usfft utils hypre
LOCDIR   = src/dm/impls/da/
MANSEC   = DM
SUBMANSEC= DMDA

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test

