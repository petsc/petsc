#PETSC_DIR = ../../../..
ALL: ex1

CFLAGS	        = $(CPPFLAGS) -D__DIR__='"$(LOCDIR)"'
FFLAGS	        = 
SOURCEC	        = 
SOURCEF	        =
SOURCEH	        =
OBJSC	        =
OBJSF	        =
LIBBASE	        = libpetscsys
LOCDIR          = src/draw/examples/tests/
TESTEXAMPLES_2  = ex1.PETSc runex1 ex1.rm ex2.PETSc runex2 ex2.rm ex3.PETSc \
		 runex3 ex3.rm ex4.PETSc runex4 runex4_2 ex4.rm clean
TESTEXAMPLES_3  = ex5.PETSc ex5.rm clean
TESTEXAMPLES_4  = ex1.PETSc ex1.rm ex4.PETSc ex4.rm clean
BUILDEXAMPLES_2 = ex1.PETSc ex1.rm ex2.PETSc ex2.rm ex3.PETSc \
		  ex3.rm ex4.PETSc ex4.rm clean
BUILDEXAMPLES_3 = ex5.PETSc ex5.rm clean
BUILDEXAMPLES_4 = ex1.PETSc ex1.rm ex4.PETSc ex4.rm clean
EXAMPLESC       = ex1.c ex2.c ex3.c ex4.c
EXAMPLESF       = ex5.F

runex1:
	-@$(MPIRUN) ex1
runex2:
	-@$(MPIRUN) ex2
runex3:
	-@$(MPIRUN) ex3
runex4:
	-@$(MPIRUN) ex4
runex4_2:
	-@$(MPIRUN) -np 2 ex4

ex1: ex1.o  chkopts
	-$(CLINKER) -o ex1 ex1.o  $(PETSC_LIB)
	$(RM) -f ex1.o

ex2: ex2.o  chkopts
	-$(CLINKER) -o ex2 ex2.o  $(PETSC_LIB)
	$(RM) -f ex2.o

ex3: ex3.o  chkopts
	-$(CLINKER) -o ex3 ex3.o  $(PETSC_LIB)
	$(RM) -f ex3.o

ex4: ex4.o  chkopts
	-$(CLINKER) -o ex4 ex4.o  $(PETSC_LIB)
	$(RM) -f ex4.o

ex5: ex5.o  chkopts
	-$(FLINKER) -o ex5 ex5.o $(PETSC_FORTRAN_LIB) $(PETSC_LIB)
	$(RM) -f ex5.o

ex6: ex6.o  chkopts
	-$(CLINKER) -o ex6 ex6.o  $(PETSC_LIB)
	$(RM) -f ex6.o

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
