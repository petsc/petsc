
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

matlabcodes: sread sopen sclose

CFLAGS    =
FFLAGS    =
CPPFLAGS  = 
SOURCEC   = 
SOURCEF   =
SOURCEH   = sopen.c sclose.c  bread.c sread.c
OBJSC     = 
OBJSF     =
MANSEC    = Viewer
LIBBASE   = libpetsc
DIRS      = 
LOCDIR    = src/sys/viewer/impls/socket/matlab/
NOADIC    = true

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test

sread:  
	-@${MATLAB_MEX} -g CC=${CC} CFLAGS='${COPTFLAGS} ${CFLAGS} ${CCPPFLAGS}' sread.c bread.c -lc
	-@${RM} -f sread.o bread.o
	-@if [ ${PETSC_ARCH} = IRIX64 ] ; then mv -f sread.mexsg64 sread.mexsg ; fi 
	-@${MV} sread.mex* ${PETSC_DIR}/bin/matlab

sopen:  
	-@${MATLAB_MEX} -g CC=${CC} CFLAGS='${COPTFLAGS} ${CFLAGS} ${CCPPFLAGS}' sopen.c -lc
	-@${RM} -f sopen.o
	-@if [ ${PETSC_ARCH} = IRIX64 ] ; then mv -f sopen.mexsg64 sopen.mexsg ; fi 
	-@${MV} sopen.mex* ${PETSC_DIR}/bin/matlab

sclose:  
	-@${MATLAB_MEX} -g CC=${CC} CFLAGS='${COPTFLAGS} ${CFLAGS} ${CCPPFLAGS}' sclose.c -lc
	-@${RM} -f sclose.o
	-@if [ ${PETSC_ARCH} = IRIX64 ] ; then mv -f sclose.mexsg64 sclose.mexsg ; fi 
	-@${MV} sclose.mex* ${PETSC_DIR}/bin/matlab

