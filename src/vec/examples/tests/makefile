ITOOLSDIR = ../..
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
EXAMPLES = ex1 ex2 ex3 ex4
EXAMPLESC= ex1.c ex2.c ex3.c ex4.c
SYSLIB   = $(LDIR)/libpetscvec.a \
           $(LDIR)/libpetscsys.a $(BLAS) $(CLIB) $(SLIB) -lm 
LINCLUDE = $(SOURCEH)

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
