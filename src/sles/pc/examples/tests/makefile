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
EXAMPLES = ex1 ex2 ex3 ex4 ex5
EXAMPLESC= ex1.c ex2.c ex3.c ex4.c ex5.c
SYSLIB   = -L$(LDIR) -lpetscsles -lpetscksp -lpetscmat \
           -lpetscvec \
           -lpetscsys -lpetscdraw $(LAPACK_LIB) $(BLAS_LIB) $(X11_LIB) \
           $(MPI_LIB) $(SYS_LIB) -lm 
LINCLUDE = $(SOURCEH)

runexamples: 
	-$(MPIRUN) ex1
	-$(MPIRUN) ex2 -kspmethod cg -kspmonitor 
	-$(MPIRUN) ex3 -kspmethod gmres -kspmonitor -pcmethod sor\
                             -sor_symmetric
	-$(MPIRUN) ex4 -kspmethod cg -kspmonitor 
	-$(MPIRUN) ex4 -kspmethod bcgs -kspmonitor 

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

ex5: ex5.o 
	-$(CLINKER) -o ex5 ex5.o $(LIBNAME) $(SYSLIB)
	$(RM) ex5.o

include $(ITOOLSDIR)/bmake/$(PARCH)/$(PARCH)
