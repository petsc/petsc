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
EXAMPLES = ex1 ex2 ex3 ex4 ex5 ex6 ex7 ex8 ex9 ex10 ex11
EXAMPLESC= ex1.c ex2.c ex3.c ex4.c ex5.c ex6.c ex7.c ex8.c ex9.c ex10.c \
           ex11.c
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

include $(ITOOLSDIR)/bmake/$(PARCH)/$(PARCH)

