IPETSCDIR = ../../../
ALL: ex1

CFLAGS   = $(OPT) $(INCLUDEDIRS) $(COPT) $(CONF) \
           -D__DIR__='"options/examples/"'
SOURCEC  = 
SOURCEF  =
WSOURCEC = 
SOURCEH  =
OBJSC    =
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscvec
EXAMPLES = ex1 ex2 ex3 ex4 ex5
EXAMPLESC= ex1.c ex2.c ex3.c ex4.c ex5.c
LINCLUDE = $(SOURCEH)

runexamples:
	-@echo "Examples in src/options/examples ----------------"
	-@echo "Ignore errors of the form p*: p4_error: (null): 1"
	-@$(MPIRUN) ex1  > ex1_1.tmp 2>&1;\
           if (diff ex1_1.out ex1_1.tmp) then true; \
           else echo "Possible problem with ex1_1, diffs above"; fi;\
           $(RM) -f ex1_1.tmp
	-@$(MPIRUN) ex2 > ex2_1.tmp 2>&1;\
           if (diff ex2_1.out ex2_1.tmp) then true; \
           else echo "Possible problem with ex2_1, diffs above"; fi;\
           $(RM) -f ex2_1.tmp
	-@$(MPIRUN) ex3  > ex3_1.tmp 2>&1;\
           if (diff ex3_1.out ex3_1.tmp) then true; \
           else echo "Possible problem with ex3_1, diffs above"; fi;\
           $(RM) -f ex3_1.tmp
	-@$(MPIRUN) ex4  > ex4_1.tmp 2>&1;\
           if (diff ex4_1.out ex4_1.tmp) then true; \
           else echo "Possible problem with ex4_1, diffs above"; fi;\
           $(RM) -f ex4_1.tmp
	-@$(MPIRUN) ex5 > ex5_1.tmp 2>&1;\
           if (diff ex5_1.out ex5_1.tmp) then true; \
           else echo "Possible problem with ex5_1, diffs above"; fi;\
           $(RM) -f ex5_1.tmp
	-@$(MPIRUN) ex5 -fptrap  > ex5_2.tmp 2>&1;\
           if (diff ex5_2.out ex5_2.tmp) then true; \
           else echo "Possible problem with ex5_2, diffs above"; fi;\
           $(RM) -f ex5_2.tmp

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
