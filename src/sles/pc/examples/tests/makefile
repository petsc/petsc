PETSC_DIR = ../../..
ALL: ex1

CFLAGS	      =	  $(PCONF) $(PETSC_INCLUDE) $(BS_INCLUDE)

SOURCEC	      = 
SOURCEF	      =
SOURCEH	      =
OBJSC	      =
OBJSF	      =
LIBBASE	      = libpetscvec
EXAMPLES_1    = ex1 ex2 ex3 ex4
RUNEXAMPLES_1 = runex1 runex2 runex3 runex4
EXAMPLES_2    = ex5
RUNEXAMPLES_2 = runex5
EXAMPLES_4    = ex1 ex2 ex3 ex4 ex5
RUNEXAMPLES_4 = runex1 runex2 runex3 runex4 runex5
EXAMPLESC     = ex1.c ex2.c ex3.c ex4.c ex5.c


ex1: ex1.o 
	-$(CLINKER) -o ex1 ex1.o $(SLES_LIB)
	$(RM) ex1.o

ex2: ex2.o 
	-$(CLINKER) -o ex2 ex2.o $(SLES_LIB)
	$(RM) ex2.o

ex3: ex3.o 
	-$(CLINKER) -o ex3 ex3.o $(SLES_LIB)
	$(RM) ex3.o

ex4: ex4.o 
	-$(CLINKER) -o ex4 ex4.o $(SLES_LIB)
	$(RM) ex4.o

ex5: ex5.o 
	-$(CLINKER) -o ex5 ex5.o $(SLES_LIB)
	$(RM) ex5.o

runex1:
	-@$(MPIRUN) ex1 > ex1_1.tmp 2>&1;   \
	   if (diff output/ex1_1.out ex1_1.tmp) then true; \
	   else echo "Possible problem with ex1_1, diffs above"; fi; \
	   $(RM) -f ex1_1.tmp 
runex2:
	-@$(MPIRUN) ex2 -ksp_type cg -ksp_smonitor > ex2_1.tmp 2>&1;	 \
	   if (diff output/ex2_1.out ex2_1.tmp) then true; \
	   else echo "Possible problem with ex2_1, diffs above"; fi; \
	   $(RM) -f ex2_1.tmp 
runex3:
	-@$(MPIRUN) ex3 -ksp_type gmres -ksp_smonitor -pc_type sor\
			     -pc_sor_symmetric > ex3_1.tmp 2>&1;   \
	   if (diff output/ex3_1.out ex3_1.tmp) then true; \
	   else echo "Possible problem with ex3_1, diffs above"; fi; \
	   $(RM) -f ex3_1.tmp 
runex4:
	-@$(MPIRUN) ex4	 > ex4_1.tmp 2>&1;   \
	   if (diff output/ex4_1.out ex4_1.tmp) then true; \
	   else echo "Possible problem with ex4_1, diffs above"; fi; \
	   $(RM) -f ex4_1.tmp 

runex5:
	-@$(MPIRUN) ex5 > ex5_1.tmp 2>&1;   \
	   if (diff output/ex5_1.out ex5_1.tmp) then true; \
	   else echo "Possible problem with ex5_1, diffs above"; fi; \
	   $(RM) -f ex5_1.tmp 

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
