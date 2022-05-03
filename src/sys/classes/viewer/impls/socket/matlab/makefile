#requirespackage  'PETSC_USE_MATLAB_SOCKET'
#requiresscalar real
#requiresprecision  double
#
#   matlabcodes makes the MATLAB mex files for receiving PETSc objects
# in MATLAB. You cannot use a complex numbers.
#
matlabcodes: sread sopen sclose swrite

SOURCEC   =
SOURCEF   =
SOURCEH   = sopen.c sclose.c  bread.c sread.c swrite.c
MANSEC    = Sys
SUBMANSEC = Viewer
LIBBASE   = libpetscsys
DIRS      =
# LOCDIR is set to 0 because the string version is not passed properly down to MATLAB's mex program
LOCDIR    = src/sys/classes/viewer/impls/socket/matlab/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

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

