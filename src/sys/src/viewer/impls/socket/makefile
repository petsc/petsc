# $Id: makefile,v 1.40 1998/09/29 02:32:55 bsmith Exp bsmith $ 

#
#   matlabcodes makes the Matlab mex files for receiving PETSc objects
# in Matlab. You cannot use a BOPT=*_C++ or BOPT=*_complex. If cmex fails 
# in compiling these files and you are using the gcc compiler then try 
# removing the -fPIC from the CC declaration in ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
#
ALL: lib
matlabcodes: receive openport closeport launch

CFLAGS    =
FFLAGS    =
CPPFLAGS  = ${MATLAB_INCLUDE}
SOURCEC   = senddense.c send.c sendsparse.c
SOURCEF   =
SOURCEH   = matlab.h closeport.c openport.c receivedense.c launch.c receive.c  \
            receivesparse.c closeport.m launch.m openport.m receive.m
OBJSC     = senddense.o send.o  sendsparse.o
OBJSF     =
MANSEC    = v
LIBBASE   = libpetscsys
DIRS      = 
LOCDIR    = src/viewer/impls/matlab/
NOADIC    = true

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

# Depending on your compiler, you may need to change CC below to cc, for example.
receive:  
	${CMEX} -V4 -g CC=${MCC} ${CFLAGS} ${CCPPFLAGS} receive.c receivesparse.c  receivedense.c 
	${RM} -f receive.o receivesparse.o  receivedense.o 

openport: 
	${CMEX} -V4 -g CC=${MCC} ${CFLAGS} ${CCPPFLAGS} openport.c 
	${RM} -f  openport.o 

closeport:
	${CMEX} -V4 -g CC=${MCC} ${CFLAGS} ${CCPPFLAGS} closeport.c
	${RM} -f  closeport.o 

launch: 
	${CMEX} -V4 -g CC=${MCC}  ${CFLAGS} ${CCPPFLAGS} launch.c
	${RM} -f  launch.o 



