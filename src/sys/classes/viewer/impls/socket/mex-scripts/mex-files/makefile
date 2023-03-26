-include ../../../../../../../petscdir.mk
#requirespackage  'PETSC_USE_MATLAB_SOCKET'
#requiresscalar real
#requiresprecision  double
#
#   mex-scripts makes the MATLAB mex files for receiving PETSc objects
# in MATLAB. You cannot use a complex numbers.
#
mex-scripts: sread sopen sclose swrite

MANSEC    = Sys
SUBMANSEC = Viewer
LIBBASE   = libpetscsys

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules.doc

sread:
	-@${MATLAB_MEX}  -g GCC='${CC}' CC='${PCC}' CFLAGS='${COPTFLAGS} ${CC_FLAGS} ${CCPPFLAGS}' LDFLAGS='${PETSC_EXTERNAL_LIB_BASIC}' sread.c bread.c
	-@${RM} -f sread.o bread.o
	-@${MV} sread.mex* ${PETSC_DIR}/${PETSC_ARCH}/lib/petsc/matlab

swrite:
	-@${MATLAB_MEX} -g GCC='${CC}' CC='${PCC}' CFLAGS='${COPTFLAGS} ${CC_FLAGS} ${CCPPFLAGS}' LDFLAGS='${PETSC_EXTERNAL_LIB_BASIC}' swrite.c bread.c
	-@${RM} -f swrite.o bread.o
	-@${MV} swrite.mex* ${PETSC_DIR}/${PETSC_ARCH}/lib/petsc/matlab

sopen:
	-@${MATLAB_MEX} -g GCC='${CC}' CC='${PCC}' CFLAGS='${COPTFLAGS} ${CC_FLAGS} ${CCPPFLAGS} ${PCC_FLAGS}' LDFLAGS='${PETSC_EXTERNAL_LIB_BASIC}' sopen.c
	-@${RM} -f sopen.o
	-@${MV} sopen.mex* ${PETSC_DIR}/${PETSC_ARCH}/lib/petsc/matlab

sclose:
	-@${MATLAB_MEX} -g GCC='${CC}' CC='${PCC}' CFLAGS='${COPTFLAGS} ${CC_FLAGS} ${CCPPFLAGS} ${PCC_FLAGS}' LDFLAGS='${PETSC_EXTERNAL_LIB_BASIC}' sclose.c
	-@${RM} -f sclose.o
	-@${MV} sclose.mex* ${PETSC_DIR}/${PETSC_ARCH}/lib/petsc/matlab

