ITOOLSDIR = ../../../
ALL: ex1

CFLAGS   = $(OPT)  $(CONF) $(INCLUDEDIRS)
SOURCEC  = 
SOURCEF  =
WSOURCEC = 
SOURCEH  =
OBJSC    =
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscvec
EXAMPLES = ex1 ex2 ex3 ex4
EXAMPLESC= ex1.c ex2.c ex3.c ex4.c
SYSLIB   = -L$(LDIR) -lpetscpc -lpetscsles -lpetscksp -lpetscmat \
           -lpetscvec \
           -lpetscsys -lpetscdraw $(LAPACK) $(BLAS) $(X11LIB) $(CLIB) $(SLIB) -lm 
LINCLUDE = $(SOURCEH)

runexamples: 
	-ex1
	-ex2 -kspmethod cg -kspmonitor 
	-ex3 -kspmethod gmres -kspmonitor -pcmethod sor -sor_symmetric

ex1: ex1.o 
	-$(CLINKER) -o ex1 ex1.o $(LIBNAME) $(SYSLIB)
	$(RM) ex1.o

ex2: ex2.o 
	-$(CLINKER) -o ex2 ex2.o $(LIBNAME) $(SYSLIB)
	$(RM) ex2.o

ex3: ex3.o 
	-$(CLINKER) -o ex3 ex3.o $(LIBNAME) $(SYSLIB)
	$(RM) ex3.o

ex4: ex4.o 
	-$(CLINKER) -o ex4 ex4.o $(LIBNAME) $(SYSLIB)
	$(RM) ex4.o

include $(ITOOLSDIR)/bmake/$(PARCH)/$(PARCH)
