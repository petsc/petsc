ITOOLSDIR = ../..
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
EXAMPLES = ex1 ex2 ex4 ex5 ex6 ex7 ex8 ex9
EXAMPLESC= ex1.c ex2.c ex4.c ex5.c ex6.c ex7.c ex8.c ex9.c
SYSLIB   = $(LDIR)/libpetscvec.a $(LDIR)/libpetscsys.a \
           $(LAPACK) $(BLAS) \
           $(CLIB) $(SLIB) $(FCLIB) -lm 
LINCLUDE = $(SOURCEH)

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

include $(ITOOLSDIR)/bmake/$(PARCH)/$(PARCH)
