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
EXAMPLES  = ex1 ex2 ex3 ex4 ex5 ex6
EXAMPLESC = ex1.c ex2.c ex3.c ex4.c ex5.c ex6.c
SYSLIB    = -L$(LDIR) -lpetscstencil -lpetscgrid \
            -lpetscsles -lpetscksp -lpetscmat \
           -lpetscvec -lpetscsys -lpetscdraw \
           $(LAPACK_LIB) $(BLAS_LIB) $(X11_LIB) $(MPI_LIB) $(SYS_LIB) \
           $(FC_LIB) -lm 
LINCLUDE  = $(SOURCEH)

runexamples:
	-$(MPIRUN) ex1 -kspmonitor
	-$(MPIRUN) -np 2 ex1  -kspmonitor
	-$(MPIRUN) -np 1 ex1 -pcmethod sor -sor_symmetric -kspmonitor
	-$(MPIRUN) -np 2 ex1 -pcmethod sor -sor_symmetric -kspmonitor
	-$(MPIRUN) -np 1 ex1 -pcmethod eisenstat -kspmonitor
	-$(MPIRUN) -np 2 ex1 -pcmethod eisenstat -kspmonitor
	-$(MPIRUN) ex2 -kspmonitor -m 5 
	-$(MPIRUN) -np 2 ex2 -kspmonitor -m 5 
	-$(MPIRUN) -np 1 ex2 -pcmethod sor -sor_symmetric -kspmonitor
	-$(MPIRUN) -np 2 ex2 -pcmethod sor -sor_symmetric -kspmonitor
	-$(MPIRUN) -np 1 ex2 -pcmethod eisenstat -kspmonitor
	-$(MPIRUN) -np 2 ex2 -pcmethod eisenstat -kspmonitor
	-$(MPIRUN) ex3 -kspmonitor -m 5 
	-$(MPIRUN) -np 2 ex3 -kspmonitor -m 5 
	-$(MPIRUN) -np 2 ex3 -pcmethod sor -sor_symmetric -kspmonitor
	-$(MPIRUN) -np 2 ex3 -pcmethod eisenstat -kspmonitor
	-$(MPIRUN) ex4 -kspmonitor -m 5
	-$(MPIRUN) -np 2 ex4 -kspmonitor -m 5 
	-$(MPIRUN) -np 1 ex4 -pcmethod sor -sor_symmetric -kspmonitor -m 5 
	-$(MPIRUN) -np 2 ex4 -pcmethod sor -sor_symmetric -kspmonitor -m 5 
	-$(MPIRUN) -np 1 ex4 -pcmethod eisenstat -kspmonitor -m 5 
	-$(MPIRUN) -np 2 ex4 -pcmethod eisenstat -kspmonitor -m 5 
	-$(MPIRUN) ex5 -kspmonitor -m 5 
	-$(MPIRUN) -np 2 ex5 -kspmonitor -m 5 
	-$(MPIRUN) ex6 -kspmonitor -m 5 
	-$(MPIRUN) -np 2 ex6 -kspmonitor -m 5 
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

ex6: ex6.o 
	-$(CLINKER) -o ex6 ex6.o $(LIBNAME) $(SYSLIB)
	$(RM) ex6.o

include $(ITOOLSDIR)/bmake/$(PARCH)/$(PARCH)
