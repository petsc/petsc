PETSC_DIR = ../../..
ALL: ex1

CFLAGS	      =	 $(PETSC_INCLUDE) $(PCONF) 
FFLAGS	      = 
SOURCEC	      = 
SOURCEF	      =
SOURCEH	      =
OBJSC	      =
OBJSF	      =
LIBBASE	      = libpetscvec
RUNEXAMPLES_2 = 
# temporarily zap this because it is hanging on the nightly builds
# RUNEXAMPLES_2 = runex1
EXAMPLES_2    = ex1 ex2 ex3 ex4 ex5 ex6
EXAMPLESC     = ex1.c ex2.c ex3.c ex4.c ex5.c ex6.c
EXAMPLES_4    = ex1 ex2 ex3 ex4 ex5 ex6
RUNEXAMPLES_4 =	 

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

ex6: ex6.o 
	-$(CLINKER) -o ex6 ex6.o  $(PETSC_LIB)
	$(RM) -f ex6.o

runex1:
	-@$(MPIRUN) -np 2 ex1 > ex1_1.tmp 2>&1;	  \
	   if (diff output/ex1_1.out ex1_1.tmp) then true; \
	   else echo "Possible problem with ex1_1, diffs above"; fi; \
	   $(RM) -f ex1_1.tmp

runex2:
	-@$(MPIRUN) -np 2 ex2 > ex2_1.tmp 2>&1;	  \
	   if (diff output/ex2_1.out ex2_1.tmp) then true; \
	   else echo "Possible problem with ex2_1, diffs above"; fi; \
	   $(RM) -f ex2_1.tmp

runex3:
	-@$(MPIRUN) -np 3 ex3 -time 50

runex4:
	-@$(MPIRUN) -np 4 ex4 > ex4_1.tmp 2>&1;	  \
	   if (diff output/ex4_1.out ex4_1.tmp) then true; \
	   else echo "Possible problem with ex4_1, diffs above"; fi; \
	   $(RM) -f ex4_1.tmp

runex5:
	-@$(MPIRUN) -np 3 ex5 -time 50

runex6:
	-@$(MPIRUN) -np 6 ex6 

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
