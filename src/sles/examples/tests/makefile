ITOOLSDIR = ../../../
ALL: ex1

CFLAGS    = $(OPT) $(CONF) $(INCLUDEDIRS)
SOURCEC   = 
SOURCEF   =
WSOURCEC  = 
SOURCEH   =
OBJSC     =
WOBJS     = 
OBJSF     =
LIBBASE   = libpetscsles
EXAMPLES  = ex1 ex2 ex3 ex4
EXAMPLESC = ex1.c ex2.c ex3.c ex4.c
SYSLIB    = -L$(LDIR) -lpetscsles -lpetscksp -lpetscmat \
           -lpetscvec -lpetscsys -lpetscdraw \
           $(LAPACK) $(BLAS) $(X11LIB) $(CLIB) $(SLIB) $(FCLIB) -lm 
LINCLUDE  = $(SOURCEH)

runexamples:
	-$(MPIRUN) ex1 
	-$(MPIRUN) -np 2 ex1 
	-$(MPIRUN) ex2 -m 5
	-$(MPIRUN) -np 2 ex2 -m 5
	-$(MPIRUN) ex3 -m 5
	-$(MPIRUN) -np 2 ex3 -m 5

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
