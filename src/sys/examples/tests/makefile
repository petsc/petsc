# $Id: makefile,v 1.17 1997/09/10 18:53:13 balay Exp balay $ 

ALL: ex1

CFLAGS	        = $(CPPFLAGS)  $(FCONF) -D__SDIR__='"$(LOCDIR)"'
FFLAGS	        = 
SOURCEC	        = 
SOURCEF	        =
SOURCEH	        =
OBJSC	        =
OBJSF	        =
LIBBASE	        = libpetscsys
LOCDIR          = src/sys/examples/tests/
TESTEXAMPLES_2  = ex1.PETSc runex1 ex1.rm ex2.PETSc runex2 ex2.rm ex3.PETSc\
		  runex3 ex3.rm ex8.PETSc runex8 ex8.rm clean
TESTEXAMPLES_3  = ex5.PETSc ex5.rm ex6.PETSc ex6.rm clean
TESTEXAMPLES_9  = ex5.PETSc ex5.rm ex6.PETSc ex6.rm clean
TESTEXAMPLES_4  = ex1.PETSc ex1.rm ex2.PETSc ex2.rm ex3.PETSc ex3.rm clean
BUILDEXAMPLES_2 = ex1.PETSc ex1.rm ex2.PETSc ex2.rm ex3.PETSc\
		  ex3.rm clean
BUILDEXAMPLES_3 = ex5.PETSc ex5.rm ex6.PETSc ex6.rm clean
BUILDEXAMPLES_9 = ex5.PETSc ex5.rm ex6.PETSc ex6.rm clean
BUILDEXAMPLES_4 = ex1.PETSc ex1.rm ex2.PETSc ex2.rm ex3.PETSc ex3.rm clean

EXAMPLESC       = ex1.c ex2.c ex3.c ex7.c ex8.c ex9.c
EXAMPLESF       = ex5.F ex6.F ex7f.F

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

ex1: ex1.o chkopts
	-$(CLINKER) -o ex1 ex1.o  $(PETSC_LIB)
	$(RM) -f ex1.o

ex2: ex2.o chkopts
	-$(CLINKER) -o ex2 ex2.o  $(PETSC_LIB)
	$(RM) -f ex2.o

ex3: ex3.o chkopts
	-$(CLINKER) -o ex3 ex3.o  $(PETSC_LIB)
	$(RM) -f ex3.o

ex4: ex4.o chkopts
	-$(CLINKER) -o ex4 ex4.o  $(PETSC_LIB)
	$(RM) -f ex4.o

ex5: ex5.o chkopts
	-$(FLINKER) -o ex5 ex5.o  $(PETSC_FORTRAN_LIB) $(PETSC_LIB)
	$(RM) -f ex5.o

ex6: ex6.o chkopts
	-$(FLINKER) -o ex6 ex6.o  $(PETSC_FORTRAN_LIB) $(PETSC_LIB)
	$(RM) -f ex6.o

ex7: ex7.o chkopts
	-$(CLINKER) -o ex7 ex7.o  $(PETSC_LIB)
	$(RM) -f ex7.o

ex7f: ex7f.o chkopts
	-$(FLINKER) -o ex7f ex7f.o  $(PETSC_FORTRAN_LIB) $(PETSC_LIB)
	$(RM) -f ex7f.o

ex8: ex8.o chkopts
	-$(CLINKER) -o ex8 ex8.o  $(PETSC_LIB)
	$(RM) -f ex8.o

ex9: ex9.o chkopts
	-$(CLINKER) -o ex9 ex9.o  $(PETSC_LIB)
	$(RM) -f ex9.o

ex10: ex10.o chkopts
	-$(CLINKER) -o ex10 ex10.o  $(PETSC_LIB)
	$(RM) -f ex10.o

ex11: ex11.o chkopts
	-$(CLINKER) -o ex11 ex11.o  $(PETSC_LIB)
	$(RM) -f ex11.o

ex12: ex12.o chkopts
	-$(CLINKER) -o ex12 ex12.o  $(PETSC_LIB)
	$(RM) -f ex12.o

runex1:
	-@$(MPIRUN) -np 1 ex1 > ex1_1.tmp 2>&1; grep -v p0 ex1_1.tmp > ex1_1.tmpz ;\
	   if (diff output/ex1_1.out ex1_1.tmpz) then true; \
	   else echo "This is a contrived error to test error handling."; fi; \
	   $(RM) -f ex1_1.tmp ex1_1.tmpz

runex2:
	-@$(MPIRUN) -np 1 ex2	 > ex2_1.tmp 2>&1;grep -v p0 ex2_1.tmp > ex2_1.tmpz ;\
	   if (diff output/ex2_1.out ex2_1.tmpz) then true; \
	   else echo "This is a contrived error to test error handling."; fi; \
	   $(RM) -f ex2_1.tmp ex2_1.tmpz

runex3:
	-@$(MPIRUN) -np 1 ex3 -fp_trap > ex3_1.tmp 2>&1;\
	   grep -v p0 ex3_1.tmp > ex3_1.tmpz ;	 \
	   if (diff output/ex3_1.out ex3_1.tmpz) then true; \
	   else echo "This is a contrived error to test floating point traps."; fi; \
	   $(RM) -f ex3_1.tmp ex3_1.tmpz

runex8:
	-@$(MPIRUN) -np 1 ex8 -trmalloc_nan -fp_trap > ex8_1.tmp 2>&1;\
	   grep -v p0 ex8_1.tmp > ex8_1.tmpz ;	 \
	   if (diff output/ex8_1.out ex8_1.tmpz) then true; \
	   else echo "This is a contrived error to test for uninitialized memory."; fi; \
	   $(RM) -f ex8_1.tmp ex8_1.tmpz

runex10:
	-@$(MPIRUN) -np 1 ex10 > ex10_1.tmp 2>&1;\
	   if (diff output/ex10_1.out ex10_1.tmp) then true; \
	   else echo "Error in PetscMemmove."; fi; \
	   $(RM) -f ex10_1.tmp ex10_1.tmp



