ITOOLSDIR = ../../..
ALL: 

CFLAGS   = $(OPT) -I$(ITOOLSDIR) $(CONF) -I$(ITOOLSDIR)/include
SOURCEC  = 
SOURCEF  =
WSOURCEC = 
SOURCEH  =
OBJSC    =
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscis
TESTS    = 
TESTSC   = 
SYSLIB   = -L$(LDIR) -lpetscvec -lpetscsys \
           $(BLAS) $(SLIB) -lm 
LINCLUDE = $(SOURCEH)

runexamples:

include $(ITOOLSDIR)/bmake/$(PARCH)/$(PARCH)
