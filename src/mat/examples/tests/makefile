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
EXAMPLES = ex1 ex2 ex3 ex4 ex5 ex6 ex7 ex8 ex9 ex10 ex11 ex12
EXAMPLESC= ex1.c ex2.c ex3.c ex4.c ex5.c ex6.c ex7.c ex8.c ex9.c\
           ex10.c ex11.c ex12.c
SYSLIB   = -L$(LDIR) -lpetscvec -lpetscsys \
           $(LAPACK) $(BLAS) \
           $(CLIB) $(SLIB) $(FCLIB) -lm 
LINCLUDE = $(SOURCEH)

runexamples:
	-ex1
	-ex2
	-ex3
	-ex4
	-ex5
	-ex6
	-ex7
	-ex8
	-mpirun $(NP) ex9
	-ex10
	-ex11
	-mpirun $(NP) ex12

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

include $(ITOOLSDIR)/bmake/$(PARCH)/$(PARCH)
