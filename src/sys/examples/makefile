PETSC_DIR = ../../..
ALL: ex1

CFLAGS	      =	 $(PETSC_INCLUDE) $(COPT) $(PCONF) \
		 -D__DIR__='"sys/examples/"'
FFLAGS	      = 
SOURCEC	      = 
SOURCEF	      =
SOURCEH	      =
OBJSC	      =
OBJSF	      =
LIBBASE	      = libpetscsys
RUNEXAMPLES_2 = runex1 runex2 runex3 runex4
EXAMPLES_2    = ex1 ex2 ex3 ex4
RUNEXAMPLES_3 = 
EXAMPLES_3    = ex5
RUNEXAMPLES_4 = 
EXAMPLES_4    = ex1 ex2 ex3 ex4
EXAMPLESC     = ex1.c ex2.c ex3.c ex4.c
EXAMPLESF     = ex5.F

runex1:
	-@$(MPIRUN) ex1 > ex1_1.tmp 2>&1; grep -v p0 ex1_1.tmp > ex1_1.tmpz ;\
	   if (diff output/ex1_1.out ex1_1.tmpz) then true; \
	   else echo "Possible problem with ex1_1, diffs above"; fi; \
	   $(RM) -f ex1_1.tmp ex1_1.tmpz

runex2:
	-@$(MPIRUN) ex2	 > ex2_1.tmp 2>&1;grep -v p0 ex2_1.tmp > ex2_1.tmpz ;\
	   if (diff output/ex2_1.out ex2_1.tmpz) then true; \
	   else echo "Possible problem with ex2_1, diffs above"; fi; \
	   $(RM) -f ex2_1.tmp ex2_1.tmpz

runex3:
	-@$(MPIRUN) ex3 -fp_trap > ex3_1.tmp 2>&1;\
	   grep -v p0 ex3_1.tmp > ex3_1.tmpz ;	 \
	   if (diff output/ex3_1.out ex3_1.tmpz) then true; \
	   else echo "Possible problem with ex3_1, diffs above"; fi; \
	   $(RM) -f ex3_1.tmp ex3_1.tmpz

runex4:
	-@$(MPIRUN) ex4 

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
	-$(FLINKER) -o ex5 ex5.o  $(PETSC_FORTRAN_LIB) $(PETSC_LIB)
	$(RM) -f ex5.o

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
