PETSC_DIR = ../../../..


ALL: lib 
CFLAGS    = $(CPPFLAGS) $(CONF) -D__DIR__='"$(LOCDIR)"' \
            -I../.. $(MATLAB_INCLUDE)
SOURCEC   = senddense.c send.c  sendsparse.c
SOURCEF   =
SOURCEH   = matlab.h
OBJSC     = senddense.o send.o  sendsparse.o
OBJSF     =
MANSEC    = v
LIBBASE   = libpetscsys
DIRS      = 
LOCDIR    = src/viewer/impls/matlab/

receive: chkopts openport closeport receive.o receivesparse.o receivedense.o
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
  
include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
