IPETSCDIR = ../../../
ALL: ex1

CFLAGS    = $(OPT) $(CONF) $(INCLUDEDIRS)
SOURCEC   = 
SOURCEF   =
WSOURCEC  = 
SOURCEH   =
OBJSC     =
WOBJS     = 
OBJSF     =
LIBBASE   = libpetscsnes
EXAMPLES  = ex1 ex2 ex3
EXAMPLESC = ex1.c ex2.c ex3.c
SYSLIB    = $(PETSC_LIB)
LINCLUDE  = $(SOURCEH)


runexamples:
	-@echo "Examples in src/sles/examples -----------------"
	-@$(MPIRUN) ex1 > ex1_1.tmp 2>&1;   \
           if (diff ex1_1.out ex1_1.tmp) then true; \
           else echo "Possible problem with ex1_1, diffs above"; fi; \
           $(RM) -f ex1_1.tmp
	-@$(MPIRUN) ex1 -snes_method tr > ex1_2.tmp 2>&1;   \
           if (diff ex1_2.out ex1_2.tmp) then true; \
           else echo "Possible problem with ex1_2, diffs above"; fi; \
           $(RM) -f ex1_2.tmp

ex1: ex1.o 
	-$(CLINKER) -o ex1 ex1.o $(SYSLIB)
	$(RM) ex1.o

ex2: ex2.o 
	-$(CLINKER) -o ex2 ex2.o $(SYSLIB)
	$(RM) ex2.o

ex3: ex3.o 
	-$(CLINKER) -o ex3 ex3.o $(SYSLIB)
	$(RM) ex3.o

ex4: ex4.o 
	-$(CLINKER) -o ex4 ex4.o $(SYSLIB)
	$(RM) ex4.o

ex5: ex5.o 
	-$(CLINKER) -o ex5 ex5.o $(SYSLIB)
	$(RM) ex5.o

ex6: ex6.o 
	-$(CLINKER) -o ex6 ex6.o $(SYSLIB)
	$(RM) ex6.o

ex7: ex7.o 
	-$(CLINKER) -o ex7 ex7.o $(SYSLIB)
	$(RM) ex7.o

include $(IPETSCDIR)/bmake/$(PARCH)/$(PARCH)
