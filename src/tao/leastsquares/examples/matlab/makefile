#
#requirespackage  'PETSC_USE_MATLAB_SOCKET'
#requiresscalar real
#requiresprecision  double
#
#   matlabcodes makes the MATLAB mex files for receiving PETSc objects
# in MATLAB. You cannot use a complex numbers.
#
matlabcodes: taopounders

CFLAGS    =
FFLAGS    =
CPPFLAGS  =
SOURCEC   =
SOURCEF   =
SOURCEH   = taopounders.c
MANSEC    = Sys
SUBMANSEC = Viewer
LIBBASE   = libpetscsys
DIRS      =
# LOCDIR is set to 0 because the string version is not passed properly down to MATLAB's mex program
LOCDIR    = src/tao/leastsquares/examples/matlab

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

taopounders:
#	-@${MATLAB_MEX}  -g GCC='${CC}' CC='${PCC}' CFLAGS='${COPTFLAGS} ${CC_FLAGS} ${CCPPFLAGS}' LDFLAGS='${PETSC_EXTERNAL_LIB_BASIC}' taopounders.c -lpetsc
	-@${MATLAB_MEX}  -g GCC='${CC}' CC='${PCC}' CFLAGS='${COPTFLAGS} ${CC_FLAGS} ${CCPPFLAGS}' LDFLAGS='${PETSC_TAO_LIB}' taopounders.c 
	-@${RM} -f taopounders.o

