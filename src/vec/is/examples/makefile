PETSC_DIR = ../../..
ALL: ex1

CFLAGS        =  $(PETSC_INCLUDE) $(PCONF) 
SOURCEC       = 
SOURCEF       =
SOURCEH       =
OBJSC         =
OBJSF         =
LIBBASE       = libpetscvec
RUNEXAMPLES_1 = runex2
EXAMPLES_1    = ex2
RUNEXAMPLES_2 = runex1
EXAMPLES_2    = ex1
EXAMPLESC     = ex1.c ex2.c

runex2:
	-@$(MPIRUN)  ex2  > ex2_1.tmp 2>&1;\
           if (diff output/ex2_1.out ex2_1.tmp) then true; \
           else echo "Possible problem with ex2_1, diffs above"; fi ;\
           $(RM) -f ex2_1.tmp

runex1:
	-@$(MPIRUN)  ex1  > ex1_1.tmp 2>&1;\
           if (diff output/ex1_1.out ex1_1.tmp) then true; \
           else echo "Possible problem with ex1_1, diffs above"; fi ;\
           $(RM) -f ex1_1.tmp

ex1: ex1.o chkopts
	-$(CLINKER) -o ex1 ex1.o  $(SLES_LIB)
	$(RM) -f ex1.o

ex2: ex2.o chkopts
	-$(CLINKER) -o ex2 ex2.o  $(SLES_LIB)
	$(RM) -f ex2.o

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
