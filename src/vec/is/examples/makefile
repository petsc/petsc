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
SYSLIB   = $(LDIR)/libpetscvec.a $(LDIR)/libpetscsys.a \
           $(BLAS) $(SLIB) -lm 
LINCLUDE = $(SOURCEH)

include $(ITOOLSDIR)/bmake/$(PARCH)/$(PARCH)
