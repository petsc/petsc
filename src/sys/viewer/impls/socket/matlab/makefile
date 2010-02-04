
#
#requirespackage  'PETSC_HAVE_MATLAB'
#requiresscalar real 
#requiresprecision  double
#
#   matlabcodes makes the Matlab mex files for receiving PETSc objects
# in Matlab. You cannot use a complex numbers.
#
SPECIALLIB     = yes
SPECIALFASTLIB = yes
speciallib:
	-@echo "BEGINNING TO COMPILE MATLAB INTERFACE"
	-@if [ ! -d "${PETSC_DIR}/bin/matlab" ] ; then ${MKDIR}  ${PETSC_DIR}/bin/matlab; fi
	-@${OMAKE}  matlabcodes PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR}
specialfastlib: speciallib

matlabcodes: sread sopen sclose swrite

CFLAGS    =
FFLAGS    =
CPPFLAGS  = 
SOURCEC   = 
SOURCEF   =
SOURCEH   = sopen.c sclose.c  bread.c sread.c swrite.c
OBJSC     = 
OBJSF     =
MANSEC    = Viewer
LIBBASE   = libpetscsys
DIRS      =
# LOCDIR is set to 0 because the string version is not passed properly down to MATLAB's mex program 
LOCDIR    = 0
NOADIC    = true

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test

sread:  
	-@${MATLAB_MEX}  -g CC=${CC} CFLAGS='${COPTFLAGS} ${CC_FLAGS} ${CCPPFLAGS}' sread.c bread.c -lc
	-@${RM} -f sread.o bread.o
	-@if [ ${PETSC_ARCH} = IRIX64 ] ; then mv -f sread.mexsg64 sread.mexsg ; fi 
	-@${MV} sread.mex* ${PETSC_DIR}/bin/matlab

swrite:  
	-@${MATLAB_MEX} -g CC=${CC} CFLAGS='${COPTFLAGS} ${CC_FLAGS} ${CCPPFLAGS}' swrite.c bread.c -lc
	-@${RM} -f swrite.o bread.o
	-@if [ ${PETSC_ARCH} = IRIX64 ] ; then mv -f swrite.mexsg64 swrite.mexsg ; fi 
	-@${MV} swrite.mex* ${PETSC_DIR}/bin/matlab

sopen:  
	-@${MATLAB_MEX} -g CC=${CC} CFLAGS='${COPTFLAGS} ${CC_FLAGS} ${CCPPFLAGS} ${PCC_FLAGS}' sopen.c -lc
	-@${RM} -f sopen.o
	-@if [ ${PETSC_ARCH} = IRIX64 ] ; then mv -f sopen.mexsg64 sopen.mexsg ; fi 
	-@${MV} sopen.mex* ${PETSC_DIR}/bin/matlab

sclose:  
	-@${MATLAB_MEX} -g CC=${CC} CFLAGS='${COPTFLAGS} ${CC_FLAGS} ${CCPPFLAGS} ${PCC_FLAGS}' sclose.c -lc
	-@${RM} -f sclose.o
	-@if [ ${PETSC_ARCH} = IRIX64 ] ; then mv -f sclose.mexsg64 sclose.mexsg ; fi 
	-@${MV} sclose.mex* ${PETSC_DIR}/bin/matlab

