IPETSCDIR = ../../..
ALL: ex1

CFLAGS        = $(OPT) $(INCLUDEDIRS) $(CONF) 
SOURCEC       = 
SOURCEF       =
WSOURCE       = 
SOURCEH       =
OBJSC         =
WOBJS         = 
OBJSF         =
LIBBASE       = libpetscvec
RUNEXAMPLES_2 =
EXAMPLES_2    = ex1 ex2 ex3 ex4 ex5
EXAMPLESC     = ex1.c ex2.c ex3.c ex4.c ex5.c

ex1: ex1.o 
	-$(CLINKER) -o ex1 ex1.o  $(PETSC_LIB)
	$(RM) -f ex1.o

ex2: ex2.o 
	-$(CLINKER) -o ex2 ex2.o  $(PETSC_LIB)
	$(RM) -f ex2.o

ex3: ex3.o 
	-$(CLINKER) -o ex3 ex3.o  $(PETSC_LIB)
	$(RM) -f ex3.o

ex4: ex4.o 
	-$(CLINKER) -o ex4 ex4.o  $(PETSC_LIB)
	$(RM) -f ex4.o

ex5: ex5.o 
	-$(CLINKER) -o ex5 ex5.o  $(PETSC_LIB)
	$(RM) -f ex5.o

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
