IPETSCDIR = ../../..
ALL: ex1

CFLAGS   = $(OPT) $(INCLUDEDIRS) $(CONF) 
SOURCEC  = 
SOURCEF  =
WSOURCEC = 
SOURCEH  =
OBJSC    =
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscis
EXAMPLES = ex1 ex2
EXAMPLESC= ex1.c ex2.c
SYSLIB   = -L$(LDIR) -lpetscvec -lpetscsys \
           $(BLAS_LIB) $(MPI_LIB) $(SYS_LIB) -lm 
LINCLUDE = $(SOURCEH)

runexamples:
	-@echo "Examples in src/is/examples -----------"
	-@$(MPIRUN)  -np 4 ex1 > ex1_1.tmp 2>&1;   \
           if (diff ex1_1.out ex1_1.tmp) then true; \
           else echo "Possible problem with ex1_1, diffs above"; fi; \
           $(RM) -f ex1_1.tmp
	-@$(MPIRUN)  ex2  > ex2_1.tmp 2>&1;\
           if (diff ex2_1.out ex2_1.tmp) then true; \
           else echo "Possible problem with ex2_1, diffs above"; fi ;\
           $(RM) -f ex2_1.tmp
ex1: ex1.o 
	-$(CLINKER) -o ex1 ex1.o  $(SYSLIB)
	$(RM) -f ex1.o

ex2: ex2.o 
	-$(CLINKER) -o ex2 ex2.o  $(SYSLIB)
	$(RM) -f ex2.o

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
