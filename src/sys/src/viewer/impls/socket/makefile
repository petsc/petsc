# $Id: makefile,v 1.38 1998/06/12 21:45:04 balay Exp balay $ 

ALL: lib 

#
#   matlabcodes makes the Matlab mex files for receiving PETSc objects
# in Matlab. You cannot use a BOPT=*_C++ or BOPT=*_complex. If cmex fails 
# in compiling these files and you are using the gcc compiler then try 
# removing the -fPIC from the CC declaration in ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
#
matlabcodes: receive openport closeport launch

CFLAGS    =
FFLAGS    =
CPPFLAGS  = ${MATLAB_INCLUDE}
SOURCEC   = senddense.c send.c sendsparse.c
SOURCEF   =
SOURCEH   = matlab.h closeport.c openport.c receivedense.c launch.c receive.c receivesparse.c \
            closeport.m     launch.m        openport.m      receive.m
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
	${CMEX} -V4 -g CC=${MCC} ${CFLAGS} receive.c receivesparse.c  receivedense.c 
	${RM} -f receive.o receivesparse.o  receivedense.o 

openport: 
	${CMEX} -V4 -g CC=${MCC} ${CFLAGS} openport.c 
	${RM} -f  openport.o 

closeport:
	${CMEX} -V4 -g CC=${MCC} ${CFLAGS} closeport.c
	${RM} -f  closeport.o 

launch: 
	${CMEX} -V4 -g CC=${MCC}  ${CFLAGS} launch.c
	${RM} -f  launch.o 


