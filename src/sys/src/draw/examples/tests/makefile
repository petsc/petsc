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
LIBBASE       = libpetscdraw

RUNEXAMPLES_2 = runex1 runex2 runex3 runex4
EXAMPLES_2    = ex1 ex2 ex3 ex4
EXAMPLESC     = ex1.c ex2.c ex3.c ex4.c

runex1:
	-@$(MPIRUN) ex1
runex2:
	-@$(MPIRUN) ex2
runex3:
	-@$(MPIRUN) ex3
runex4:
	-@$(MPIRUN) ex4

ex1: ex1.o  chkopts
	-$(CLINKER) -o ex1 ex1.o  $(PETSC_LIB)
	$(RM) -f ex1.o

ex2: ex2.o  chkopts
	-$(CLINKER) -o ex2 ex2.o  $(PETSC_LIB)
	$(RM) -f ex2.o

ex3: ex3.o  chkopts
	-$(CLINKER) -o ex3 ex3.o  $(PETSC_LIB)
	$(RM) -f ex3.o

ex4: ex4.o  chkopts
	-$(CLINKER) -o ex4 ex4.o  $(PETSC_LIB)
	$(RM) -f ex4.o

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
