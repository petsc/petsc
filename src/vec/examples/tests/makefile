ITOOLSDIR = ../../../
ALL: ex1

CFLAGS   = $(OPT) $(INCLUDEDIRS) $(COPT) $(CONF) -I$(ITOOLSDIR)/include
SOURCEC  = 
SOURCEF  =
WSOURCEC = 
SOURCEH  =
OBJSC    =
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscvec
EXAMPLES = ex1 ex2 ex3 ex4 ex5 ex6 ex7 ex8 ex9 ex10 ex11 ex12 ex13 ex14 \
           ex15 ex16
EXAMPLESC= ex1.c ex2.c ex3.c ex4.c ex5.c ex6.c ex7.c ex8.c ex9.c ex10.c \
           ex11.c ex12.c ex13.c ex14.c ex15.c ex16.c
SYSLIB   = -L$(LDIR) -lpetscvec \
           -lpetscsys $(BLAS_LIB) -lpetscdraw $(X11_LIB) $(MPI_LIB) \
           $(SYS_LIB) -lm 
LINCLUDE = $(SOURCEH)

runexamples:
	-$(MPIRUN)  ex1
	-$(MPIRUN)  ex1 -mpi_vecs
	-$(MPIRUN)  -np 2 ex1 
	-$(MPIRUN)  ex2
	-$(MPIRUN) -np 2 ex3 -mpi
	-$(MPIRUN) -np 2 ex4 -mpi
	-$(MPIRUN) -np 2 ex5 -mpi
	-$(MPIRUN)  ex6
	-$(MPIRUN)  ex7
	-$(MPIRUN)  ex8
	-$(MPIRUN)  -np 2 ex9 
	-$(MPIRUN)  -np 2 ex10 
	-$(MPIRUN)  -np 2 ex11
	-$(MPIRUN)  -np 2 ex12
	-$(MPIRUN)  -np 2 ex13
	-$(MPIRUN)  -np 2 ex14
	-$(MPIRUN)  ex15
	-$(MPIRUN) -np 3 ex16
	
ex1: ex1.o 
	-$(CLINKER) -o ex1 ex1.o  $(SYSLIB)
	$(RM) -f ex1.o

ex2: ex2.o 
	-$(CLINKER) -o ex2 ex2.o  $(SYSLIB)
	$(RM) -f ex2.o

ex3: ex3.o 
	-$(CLINKER) -o ex3 ex3.o  $(SYSLIB)
	$(RM) -f ex3.o

ex4: ex4.o 
	-$(CLINKER) -o ex4 ex4.o  $(SYSLIB)
	$(RM) -f ex4.o

ex5: ex5.o 
	-$(CLINKER) -o ex5 ex5.o  $(SYSLIB)
	$(RM) -f ex5.o

ex6: ex6.o 
	-$(CLINKER) -o ex6 ex6.o  $(SYSLIB)
	$(RM) -f ex6.o

ex7: ex7.o 
	-$(CLINKER) -o ex7 ex7.o  $(SYSLIB)
	$(RM) -f ex7.o

ex8: ex8.o 
	-$(CLINKER) -o ex8 ex8.o  $(SYSLIB)
	$(RM) -f ex8.o

ex9: ex9.o 
	-$(CLINKER) -o ex9 ex9.o  $(SYSLIB)
	$(RM) -f ex9.o

ex10: ex10.o 
	-$(CLINKER) -o ex10 ex10.o  $(SYSLIB)
	$(RM) -f ex10.o

ex11: ex11.o 
	-$(CLINKER) -o ex11 ex11.o  $(SYSLIB)
	$(RM) -f ex11.o

ex12: ex12.o 
	-$(CLINKER) -o ex12 ex12.o  $(SYSLIB)
	$(RM) -f ex12.o

ex13: ex13.o 
	-$(CLINKER) -o ex13 ex13.o  $(SYSLIB)
	$(RM) -f ex13.o

ex14: ex14.o 
	-$(CLINKER) -o ex14 ex14.o  $(SYSLIB)
	$(RM) -f ex14.o

ex15: ex15.o 
	-$(CLINKER) -o ex15 ex15.o  $(SYSLIB) 
	$(RM) -f ex15.o

ex16: ex16.o 
	-$(CLINKER) -o ex16 ex16.o  $(SYSLIB) 
	$(RM) -f ex16.o

include $(ITOOLSDIR)/bmake/$(PARCH)/$(PARCH)

