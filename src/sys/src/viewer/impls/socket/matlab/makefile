# $Id: makefile,v 1.3 1999/09/03 20:48:56 bsmith Exp bsmith $ 

#
#requirespackage  '-DPETSC_HAVE_MATLAB'
#requiresscalar real 
#
#   matlabcodes makes the Matlab mex files for receiving PETSc objects
# in Matlab. You cannot use a BOPT=*_C++ or BOPT=*_complex. If cmex fails 
# in compiling these files and you are using the gcc compiler then try 
# removing the -fPIC from the CC declaration in ${PETSC_DIR}/bmake/${PETSC_ARCH}/base_variables
#
ALL: lib
matlabcodes: receive openport closeport launch readinvecs

CFLAGS    =
FFLAGS    =
CPPFLAGS  = 
SOURCEC   = 
SOURCEF   =
SOURCEH   = closeport.c openport.c receivedense.c launch.c receive.c  bread.c \
            receivesparse.c closeport.m launch.m openport.m receive.m readinvecs.c
OBJSC     = 
OBJSF     =
MANSEC    = Viewer
LIBBASE   = libpetsc
DIRS      = 
LOCDIR    = src/sys/src/viewer/impls/socket/matlab
NOADIC    = true

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

# CMEX and MCC are defined in bmake/${PETSC_ARCH}/base.site
#
receive:  
	${CMEX} -V4 -g CC=${MCC} ${CFLAGS} ${CCPPFLAGS} receive.c receivesparse.c  receivedense.c bread.c
	${RM} -f receive.o receivesparse.o  receivedense.o bread.o
	if [ ${PETSC_ARCH} = IRIX64 ] ; then mv -f receive.mexsg64 receive.mexsg ; fi 

openport: 
	${CMEX} -V4 -g CC=${MCC} ${CFLAGS} ${CCPPFLAGS} openport.c 
	${RM} -f  openport.o 
	if [ ${PETSC_ARCH} = IRIX64 ] ; then mv -f openport.mexsg64 openport.mexsg ; fi 

closeport:
	${CMEX} -V4 -g CC=${MCC} ${CFLAGS} ${CCPPFLAGS} closeport.c
	${RM} -f  closeport.o 
	if [ ${PETSC_ARCH} = IRIX64 ] ; then mv -f closeport.mexsg64 closeport.mexsg ; fi 

launch: 
	${CMEX} -V4 -g CC=${MCC}  ${CFLAGS} ${CCPPFLAGS} launch.c
	${RM} -f  launch.o 
	if [ ${PETSC_ARCH} = IRIX64 ] ; then mv -f launch.mexsg64 launch.mexsg ; fi 

readinvecs:  
	${CMEX} -V4 -g CC=${MCC} ${CFLAGS} ${CCPPFLAGS} readinvecs.c bread.c
	${RM} -f readinvecs.o bread.o
	if [ ${PETSC_ARCH} = IRIX64 ] ; then mv -f readinvecs.mexsg64 readinvecs.mexsg ; fi 

