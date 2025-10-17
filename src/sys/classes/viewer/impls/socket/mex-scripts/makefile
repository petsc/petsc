-include ../../../../../../../petscdir.mk
#requirespackage  'PETSC_USE_MATLAB_SOCKET'
#requiresscalar real
#requiresprecision  double
#
#   mex-scripts makes the MATLAB mex files for receiving PETSc objects from sockets or files
#
mex-scripts: sread sopen sclose swrite

MANSEC    = Sys
SUBMANSEC = Viewer

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules_doc.mk

sread:
	-@${MATLAB_MEX}  -g GCC='${CC}' CC='${PCC}' CFLAGS='${COPTFLAGS} ${CC_FLAGS} ${CCPPFLAGS}'  sread.c bread.c
	-@${RM} -f sread.o bread.o
	-@${MV} sread.mex* ${PETSC_DIR}/${PETSC_ARCH}/lib/petsc/matlab

swrite:
	-@${MATLAB_MEX} -g GCC='${CC}' CC='${PCC}' CFLAGS='${COPTFLAGS} ${CC_FLAGS} ${CCPPFLAGS}'  swrite.c bread.c
	-@${RM} -f swrite.o bread.o
	-@${MV} swrite.mex* ${PETSC_DIR}/${PETSC_ARCH}/lib/petsc/matlab

sopen:
	-@${MATLAB_MEX} -g GCC='${CC}' CC='${PCC}' CFLAGS='${COPTFLAGS} ${CC_FLAGS} ${CCPPFLAGS} ${PCC_FLAGS}' sopen.c
	-@${RM} -f sopen.o
	-@${MV} sopen.mex* ${PETSC_DIR}/${PETSC_ARCH}/lib/petsc/matlab

sclose:
	-@${MATLAB_MEX} -g GCC='${CC}' CC='${PCC}' CFLAGS='${COPTFLAGS} ${CC_FLAGS} ${CCPPFLAGS} ${PCC_FLAGS}' sclose.c
	-@${RM} -f sclose.o
	-@${MV} sclose.mex* ${PETSC_DIR}/${PETSC_ARCH}/lib/petsc/matlab

