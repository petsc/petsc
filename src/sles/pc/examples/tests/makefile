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
EXAMPLES = ex1 ex2 ex3
EXAMPLESC= ex1.c ex2.c ex3.c
SYSLIB   = $(LDIR)/libpetscpc.a $(LDIR)/libpetscksp.a $(LDIR)/libpetscmat.a \
           $(LDIR)/libpetscvec.a \
           $(LDIR)/libpetscsys.a $(LAPACK) $(BLAS) $(CLIB) $(SLIB) -lm 
LINCLUDE = $(SOURCEH)

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
