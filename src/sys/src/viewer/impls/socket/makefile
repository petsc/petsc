ITOOLSDIR = ../../../../


ALL: lib 
CFLAGS    = $(OPT) -I../.. $(INCLUDEDIRS) -I$(ITOOLSDIR)/pinclude \
            $(CONF) -D__DIR__='"viewer/impls/matlab/"' \
            $(MATLAB_INCLUDE)
SOURCEC   = senddense.c send.c  sendsparse.c
SOURCEF   =
SOURCEH   =
OBJSC     = senddense.o send.o  sendsparse.o
OBJSF     =
WSOURCEC  =   
WOBJS     = 
MANSEC    = x
LINCLUDE  = $(SOURCEH)
LIBBASE   = libpetscsys
DIRS      = 

runexamples:

receive: openport closeport receive.o receivesparse.o  receivedense.o
	$(CMEX) receive.o receivesparse.o  receivedense.o $(MATLABLIB)
	$(RM) -f receive.o receivesparse.o  receivedense.o 

openport: openport.o 
	$(CMEX)  openport.o $(MATLABLIB)
	$(RM) -f  openport.o 
  
closeport: closeport.o 
	$(CMEX)  closeport.o $(MATLABLIB)
	$(RM) -f  closeport.o 
  
include $(ITOOLSDIR)/bmake/$(PARCH)/$(PARCH)
