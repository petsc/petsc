
ALL: lib 

#
#   matlabcodes makes the Matlab mex files for receiving PETSc objects
# in Matlab. You cannot use a BOPT=*_C++ or BOPT=*_complex. If cmex fails 
# in compiling these files and you are using the gcc compiler then try 
# removing the -fPIC from the CC declaration in $(PETSC_DIR)/bmake/$*(PETSC_ARCH)/base
#
matlabcodes: receive openport closeport launch

CFLAGS    = $(CPPFLAGS) $(CONF) -D__SDIR__='"$(LOCDIR)"' $(MATLAB_INCLUDE)
SOURCEC   = senddense.c send.c  sendsparse.c
SOURCEF   =
SOURCEH   = matlab.h
OBJSC     = senddense.o send.o  sendsparse.o
OBJSF     =
MANSEC    = v
LIBBASE   = libpetscsys
DIRS      = 
LOCDIR    = src/viewer/impls/matlab/
NOADIC    = true

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

receive: chkopts receive.o receivesparse.o receivedense.o
	$(CMEX) receive.o receivesparse.o  receivedense.o 
	$(RM) -f receive.o receivesparse.o  receivedense.o 

openport: chkopts openport.o 
	$(CMEX)  openport.o 
	$(RM) -f  openport.o 

closeport: chkopts closeport.o 
	$(CMEX)  closeport.o
	$(RM) -f  closeport.o 

launch: chkopts launch.o 
	$(CMEX)  launch.o
	$(RM) -f  launch.o 


