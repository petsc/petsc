ITOOLSDIR = ../../..
ALL: ex1

CFLAGS   = $(OPT) $(INCLUDEDIRS)$(CONF) $(COPT)
SOURCEC  = 
SOURCEF  =
WSOURCEC = 
SOURCEH  =
OBJSC    =
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscdraw
EXAMPLES = ex1 ex2 ex3 ex4
EXAMPLESC= ex1.c ex2.c ex3.c ex4.c
SYSLIB   = -L$(LDIR) -lpetscdraw -lpetscsys \
           $(LAPACK) $(BLAS) \
           $(CLIB) $(X11LIB) $(SLIB) $(MALLOCDEBUG) $(FCLIB) -lm 
LINCLUDE = $(SOURCEH)

runexamples:
	-$(MPIRUN) ex1
	-$(MPIRUN) ex2
	-$(MPIRUN) ex3
	-$(MPIRUN) ex4

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

include $(ITOOLSDIR)/bmake/$(PARCH)/$(PARCH)
