#PETSC_DIR = ../../../..
ALL: ex1

CFLAGS	        =	 $(CPPFLAGS) -D__DIR__='"$(LOCDIR)"'
FFLAGS	        = 
SOURCEC	        = 
SOURCEF	        =
SOURCEH	        =
OBJSC	        =
OBJSF	        = 
LIBBASE	        = libpetscvec
LOCDIR          = src/is/examples/tests/
TESTEXAMPLES_1  = ex1.PETSc runex1 ex1.rm ex2.PETSc runex2 ex2.rm clean
TESTEXAMPLES_2  = 
TESTEXAMPLES_3  = ex1f.PETSc runex1f ex1f.rm ex2f.PETSc runex2f ex2f.rm clean
TESTEXAMPLES_9  = 
TESTEXAMPLES_4  = 
BUILDEXAMPLES_1 = 
BUILDEXAMPLES_2 = 
BUILDEXAMPLES_3 = 
BUILDEXAMPLES_9 = 
BUILDEXAMPLES_4 = 
EXAMPLESC       = ex1.c ex2.c
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

runex1:
	-@$(MPIRUN)  ex1 

runex1f:
	-@$(MPIRUN)  ex1f 


runex2:
	-@$(MPIRUN)  ex2 

runex2f:
	-@$(MPIRUN)  ex2f 

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
