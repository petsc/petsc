
#
#requirespackage  'PETSC_HAVE_MATLAB'
#requiresscalar real 
#requiresprecision  double
#
#   matlabcodes makes the Matlab mex files for receiving PETSc objects
# in Matlab. You cannot use a complex numbers. If mex fails 
# in compiling these files and you are using the gcc compiler then try 
# removing the -fPIC from the CC declaration in ${PETSC_DIR}/bmake/${PETSC_ARCH}/variables
#
SPECIALLIB     = yes
SPECIALFASTLIB = yes
speciallib:
	-@echo "BEGINNING TO COMPILE MATLAB INTERFACE"
	-@${OMAKE}  matlabcodes PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR}
	-@if [ ! -d "${PETSC_DIR}/bin/matlab" ] ; then ${MKDIR}  ${PETSC_DIR}/bin/matlab; fi
	-@${MV} receive.mex* openport.mex* closeport.mex* readinvecs.mex* ${PETSC_DIR}/bin/matlab
specialfastlib: speciallib

matlabcodes: receive openport closeport readinvecs

CFLAGS    =
FFLAGS    =
CPPFLAGS  = 
SOURCEC   = 
SOURCEF   =
SOURCEH   = closeport.c openport.c receivedense.c receive.c  bread.c readinvecs.c 
OBJSC     = 
OBJSF     =
MANSEC    = Viewer
LIBBASE   = libpetsc
DIRS      = 
LOCDIR    = src/sys/viewer/impls/socket/matlab/
NOADIC    = true

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test

receive:  
	-@${MATLAB_MEX} -g CC=${CC} CFLAGS='${COPTFLAGS} ${CFLAGS} ${CCPPFLAGS}' receive.c receivesparse.c  receivedense.c bread.c
	-@${RM} -f receive.o receivesparse.o  receivedense.o bread.o
	-@if [ ${PETSC_ARCH} = IRIX64 ] ; then mv -f receive.mexsg64 receive.mexsg ; fi 

openport: 
	-@${MATLAB_MEX} -g CC=${CC} CFLAGS='${COPTFLAGS} ${CFLAGS} ${CCPPFLAGS}' openport.c 
	-@${RM} -f  openport.o 
	-@if [ ${PETSC_ARCH} = IRIX64 ] ; then mv -f openport.mexsg64 openport.mexsg ; fi 

closeport:
	-@${MATLAB_MEX} -g CC=${CC} CFLAGS='${COPTFLAGS} ${CFLAGS} ${CCPPFLAGS}' closeport.c
	-@${RM} -f  closeport.o 
	-@if [ ${PETSC_ARCH} = IRIX64 ] ; then mv -f closeport.mexsg64 closeport.mexsg ; fi 

readinvecs:  
	-@${MATLAB_MEX} -g CC=${CC} CFLAGS='${COPTFLAGS} ${CFLAGS} ${CCPPFLAGS}' readinvecs.c bread.c
	-@${RM} -f readinvecs.o bread.o
	-@if [ ${PETSC_ARCH} = IRIX64 ] ; then mv -f readinvecs.mexsg64 readinvecs.mexsg ; fi 

