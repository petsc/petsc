PETSC_DIR = ../../..
ALL: ex1

CFLAGS	        = $(CPPFLAGS)  -D__DIR__='"$(LOCDIR)"'
FFLAGS	        = 
SOURCEC	        = 
SOURCEF	        =
SOURCEH	        =
OBJSC	        = 
OBJSF	        =
LIBBASE	        = libpetscvec
LOCDIR          = sec/da/examples/
TESTEXAMPLES_1  = ex1.PETSc runex1 ex1.rm ex4.PETSc runex4 ex4.rm
TESTEXAMPLES_2  = ex2.PETSc runex2 ex2.rm ex3.PETSc runex3 ex3.rm ex5.PETSc\
		  runex5 ex5.rm ex6.PETSc runex6 ex6.rm ex7.PETSc ex7.rm
TESTEXAMPLES_3  =
TESTEXAMPLES_4  = ex1.PETSc ex1.rm ex2.PETSc ex2.rm ex3.PETSc\
		  ex3.rm ex4.PETSc ex4.rm ex5.PETSc ex5.rm\
		  ex6.PETSc ex6.rm
TESTEXAMPLES_5  =
BUILDEXAMPLES_1 = ex1.PETSc ex1.rm ex4.PETSc ex4.rm
BUILDEXAMPLES_2 = ex2.PETSc ex2.rm ex3.PETSc ex3.rm ex5.PETSc\
		  ex5.rm ex6.PETSc ex6.rm ex7.PETSc ex7.rm
BUILDEXAMPLES_3 =
BUILDEXAMPLES_4 = ex1.PETSc  ex1.rm ex2.PETSc ex2.rm ex3.PETSc\
		  ex3.rm ex4.PETSc ex4.rm ex5.PETSc ex5.rm\
		  ex6.PETSc ex6.rm
BUILDEXAMPLES_5 =

EXAMPLESC       = ex1.c ex2.c ex3.c ex4.c ex5.c ex6.c ex7.c
EXAMPLESF       = 

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

ex7: ex7.o 
	-$(CLINKER) -o ex7 ex7.o  $(PETSC_LIB)
	$(RM) -f ex7.o

runex1:
	-@$(MPIRUN) -np 2 ex1 -nox > ex1_1.tmp 2>&1;	  \
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
	-@$(MPIRUN) -np 4 ex4 -nox > ex4_1.tmp 2>&1;	  \
	   if (diff output/ex4_1.out ex4_1.tmp) then true; \
	   else echo "Possible problem with ex4_1, diffs above"; fi; \
	   $(RM) -f ex4_1.tmp

runex5:
	-@$(MPIRUN) -np 3 ex5 -time 50

runex6:
	-@$(MPIRUN) -np 6 ex6

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
