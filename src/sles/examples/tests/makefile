ITOOLSDIR = ../..
ALL: ex1

CFLAGS   = $(OPT) $(CONF) $(INCLUDEDIRS)
SOURCEC  = 
SOURCEF  =
WSOURCEC = 
SOURCEH  =
OBJSC    =
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscsles
EXAMPLES = ex1
SYSLIB   = $(LDIR)/libpetscsles.a $(LDIR)/libpetscksp.a \
           $(LDIR)/libpetscpc.a $(LDIR)/libpetscmat.a \
           $(LDIR)/libpetscvec.a $(LDIR)/libpetscsys.a \
           $(LAPACK) $(BLAS) $(CLIB) $(SLIB) -lm 
LINCLUDE = $(SOURCEH)

ex1: ex1.o 
	-$(CLINKER) -o ex1 ex1.o $(LIBNAME) $(SYSLIB)
	$(RM) ex1.o

include $(ITOOLSDIR)/bmake/$(PARCH)/$(PARCH)
