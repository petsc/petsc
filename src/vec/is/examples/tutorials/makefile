PETSC_DIR = ../../../..
ALL: ex1

CFLAGS	        =	 $(CPPFLAGS) -D__DIR__='"$(LOCDIR)"'
FFLAGS	        = 
SOURCEC	        = 
SOURCEF	        =
SOURCEH	        =
OBJSC	        =
OBJSF	        = 
LIBBASE	        = libpetscvec
LOCDIR          = src/is/examples/tutorial/
TESTEXAMPLES_1  = ex1.PETSc runex1 ex1.rm ex2.PETSc runex2 ex2.rm ex3.PETSc runex3 ex3.rm clean
TESTEXAMPLES_2  = 
TESTEXAMPLES_3  = ex1f.PETSc runex1f ex1f.rm ex2f.PETSc runex2f ex2f.rm clean
TESTEXAMPLES_9  = 
TESTEXAMPLES_4  =
BUILDEXAMPLES_1 = ex1.PETSc ex1.rm ex2.PETSc ex2.rm ex3.PETSc ex3.rm clean
BUILDEXAMPLES_2 = 
BUILDEXAMPLES_3 = ex1f.PETSc ex1f.rm ex2f.PETSc ex2f.rm clean
BUILDEXAMPLES_9 = 
BUILDEXAMPLES_4 = 
EXAMPLESC       = ex1.c ex2.c ex3.c
EXAMPLESF       = ex1f.F ex2f.F

ex1: ex1.o chkopts
	-$(CLINKER) -o ex1 ex1.o  $(SLES_LIB)
	$(RM) -f ex1.o

ex1f: ex1f.o chkopts
	-$(FLINKER) -o ex1f ex1f.o  $(PETSC_FORTRAN_LIB) $(SLES_LIB)
	$(RM) -f ex1f.o

ex2: ex2.o chkopts
	-$(CLINKER) -o ex2 ex2.o  $(SLES_LIB)
	$(RM) -f ex2.o

ex2f: ex2f.o chkopts
	-$(FLINKER) -o ex2f ex2f.o  $(PETSC_FORTRAN_LIB) $(SLES_LIB)
	$(RM) -f ex2f.o

ex3: ex3.o  chkopts
	-$(CLINKER) -o ex3 ex3.o  $(SLES_LIB)
	$(RM) -f ex3.o

runex1:
	-@$(MPIRUN)  ex1  > ex1.tmp 2>&1;\
	   if (diff output/ex1.out ex1.tmp) then true; \
	   else echo "Possible problem with ex1, diffs above"; fi ;\
	   $(RM) -f ex1.tmp

runex1f:
	-@$(MPIRUN)  ex1f  > ex1f.tmp 2>&1;\
	   if (diff output/ex1f.out ex1f.tmp) then true; \
	   else echo "Possible problem with ex1f, diffs above"; fi ;\
	   $(RM) -f ex1f.tmp

runex2:
	-@$(MPIRUN)  ex2  > ex2.tmp 2>&1;\
	   if (diff output/ex2.out ex2.tmp) then true; \
	   else echo "Possible problem with ex2, diffs above"; fi ;\
	   $(RM) -f ex2.tmp

runex2f:
	-@$(MPIRUN)  ex2f  > ex2f.tmp 2>&1;\
	   if (diff output/ex2f.out ex2f.tmp) then true; \
	   else echo "Possible problem with ex2f, diffs above"; fi ;\
	   $(RM) -f ex2f.tmp

runex3:
	-@$(MPIRUN)  ex3  > ex3.tmp 2>&1;\
	   if (diff output/ex3.out ex3.tmp) then true; \
	   else echo "Possible problem with ex3, diffs above"; fi ;\
	   $(RM) -f ex3.tmp

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
