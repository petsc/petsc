ITOOLSDIR = ../../../
ALL: ex1

CFLAGS   = $(OPT) $(INCLUDEDIRS)$(CONF) $(COPT)
SOURCEC  = 
SOURCEF  =
WSOURCEC = 
SOURCEH  =
OBJSC    =
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscmat
EXAMPLES = ex1 ex2 ex3 ex4 ex5 ex6 ex7 ex8 ex9 ex10 ex11 ex12 ex13 ex14\
           ex15 ex16 ex17
EXAMPLESC= ex1.c ex2.c ex3.c ex4.c ex5.c ex6.c ex7.c ex8.c ex9.c\
           ex10.c ex11.c ex12.c ex13.c ex14.c ex15.c ex16.c ex17.c
SYSLIB   = -L$(LDIR) -lpetscvec -lpetscsys -lpetscdraw\
           $(LAPACK_LIB) $(BLAS_LIB) \
           $(MPI_LIB) $(SYS_LIB) $(FC_LIB) $(X11_LIB) -lm 
LINCLUDE = $(SOURCEH)

runexamples:
	-$(MPIRUN)  ex1
	-$(MPIRUN)  ex2
	-$(MPIRUN)  ex3
	-$(MPIRUN)  ex4
	-$(MPIRUN)  ex5
	-$(MPIRUN)  ex6
	-$(MPIRUN)  ex7
	-$(MPIRUN)  ex8
	-$(MPIRUN)  -np 2 ex9
	-$(MPIRUN)  ex10
	-$(MPIRUN)  ex11
	-$(MPIRUN)  -np 2 ex12
	-$(MPIRUN)  ex13
	-$(MPIRUN)  ex14
	-$(MPIRUN)  ex15
	-$(MPIRUN)  ex16

ex1: ex1.o
	-$(CLINKER)  -o ex1 ex1.o $(LIBNAME) $(SYSLIB)
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

ex7: ex7.o
	-$(CLINKER) -o ex7 ex7.o $(LIBNAME) $(SYSLIB)
	$(RM) ex7.o

ex8: ex8.o
	-$(CLINKER) -o ex8 ex8.o $(LIBNAME) $(SYSLIB)
	$(RM) ex8.o

ex9: ex9.o
	-$(CLINKER) -o ex9 ex9.o $(LIBNAME) $(SYSLIB)
	$(RM) ex9.o

ex10: ex10.o
	-$(CLINKER) -o ex10 ex10.o $(LIBNAME) $(SYSLIB)
	$(RM) ex10.o

ex11: ex11.o
	-$(CLINKER) -o ex11 ex11.o $(LIBNAME) $(SYSLIB)
	$(RM) ex11.o

ex12: ex12.o
	-$(CLINKER) -o ex12 ex12.o $(LIBNAME) $(SYSLIB)
	$(RM) ex12.o

ex13: ex13.o
	-$(CLINKER) -o ex13 ex13.o $(LIBNAME) $(SYSLIB)
	$(RM) ex13.o

ex14: ex14.o
	-$(CLINKER) -o ex14 ex14.o $(LIBNAME) $(SYSLIB)
	$(RM) ex14.o

ex15: ex15.o
	-$(CLINKER) -o ex15 ex15.o $(LIBNAME) $(SYSLIB)
	$(RM) ex15.o

ex16: ex16.o
	-$(CLINKER) -o ex16 ex16.o $(LIBNAME) $(SYSLIB)
	$(RM) ex16.o

ex17: ex17.o
	-$(CLINKER) -o ex17 ex17.o $(LIBNAME) $(SYSLIB)
	$(RM) ex17.o

include $(ITOOLSDIR)/bmake/$(PARCH)/$(PARCH)
