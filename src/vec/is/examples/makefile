PETSC_DIR = ../../..
ALL: ex1

CFLAGS	       =	 $(CPPFLAGS) -D__DIR__='"$(LOCDIR)"'
FFLAGS	       = 
SOURCEC	       = 
SOURCEF	       =
SOURCEH	       =
OBJSC	       =
OBJSF	       =
LIBBASE	       = libpetscvec
LOCDIR         = src/is/examples
TESTEXAMPLES_1 = ex2 runex2 clean
TESTEXAMPLES_2 = ex1 runex1 clean
TESTEXAMPLES_3 = ex3 runex3 clean
TESTEXAMPLES_4 = ex1 runex1 clean ex2 runex2 clean
EXAMPLESC      = ex1.c ex2.c
EXAMPLESF      = ex3.F
EXAMPLES       = ex1 ex2 ex3

ex1: ex1.o chkopts
	-$(CLINKER) -o ex1 ex1.o  $(SLES_LIB)
	$(RM) -f ex1.o

ex2: ex2.o chkopts
	-$(CLINKER) -o ex2 ex2.o  $(SLES_LIB)
	$(RM) -f ex2.o

ex3: ex3.o  chkopts
	-$(FLINKER) -o ex3 ex3.o  $(PETSC_FORTRAN_LIB) $(SLES_LIB)
	$(RM) -f ex3.o

runex1:
	-@$(MPIRUN)  ex1  > ex1_1.tmp 2>&1;\
	   if (diff output/ex1_1.out ex1_1.tmp) then true; \
	   else echo "Possible problem with ex1_1, diffs above"; fi ;\
	   $(RM) -f ex1_1.tmp
runex2:
	-@$(MPIRUN)  ex2  > ex2_1.tmp 2>&1;\
	   if (diff output/ex2_1.out ex2_1.tmp) then true; \
	   else echo "Possible problem with ex2_1, diffs above"; fi ;\
	   $(RM) -f ex2_1.tmp
runex3:
	-@$(MPIRUN) ex3 > ex3_1.tmp 2>&1;\
	   if (diff output/ex3_1.out ex3_1.tmp) then true; \
	   else echo "Possible problem with ex3_1, diffs above"; fi;\
	   $(RM) -f ex3_1.tmp

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
