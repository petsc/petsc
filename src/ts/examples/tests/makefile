#PETSC_DIR = ../../../..
ALL: ex1

CFLAGS	        = $(CPPFLAGS) -D__DIR__='"$(LOCDIR)"' 
FFLAGS	        = 
SOURCEC	        = 
SOURCEF	        =
SOURCEH	        =
OBJSC	        =
OBJSF	        =
LIBBASE	        = libpetscts
LOCDIR          = src/ts/examples/tests/
TESTEXAMPLES_1  = ex1.PETSc  runex1  ex1.rm 
TESTEXAMPLES_2  = 
TESTEXAMPLES_3  = ex1f.PETSc runex1f ex1f.rm
TESTEXAMPLES_4  =  
BUILDEXAMPLES_1 = ex1.PETSc   ex1.rm
BUILDEXAMPLES_3 = ex1f.PETSc  ex1f.rm

EXAMPLESC       = ex1.c   
EXAMPLESF       = ex1f.F
#
#
ex1: ex1.o  chkopts
	-$(CLINKER) -o ex1 ex1.o  $(PETSC_LIB)
	$(RM) ex1.o

ex1f: ex1f.o  chkopts
	-$(FLINKER) -o ex1f ex1f.o  $(PETSC_FORTRAN_LIB) $(PETSC_LIB)
	$(RM) ex1f.o

ex2: ex2.o  chkopts
	-$(CLINKER) -o ex2 ex2.o -L$(LDIR) -lpetscts $(PETSC_LIB)
	$(RM) ex2.o

ex3: ex3.o  chkopts
	-$(CLINKER) -o ex3 ex3.o -L$(LDIR) -lpetscts $(PETSC_LIB)
	$(RM) ex3.o

ex4: ex4.o  chkopts
	-$(CLINKER) -o ex4 ex4.o -L$(LDIR) -lpetscts $(PETSC_LIB)
	$(RM) ex4.o

ex5: ex5.o  chkopts
	-$(CLINKER) -o ex5 ex5.o -L$(LDIR) -lpetscts $(PETSC_LIB)
	$(RM) ex5.o

ex6: ex6.o  chkopts
	-$(CLINKER) -o ex6 ex6.o $(PETSC_LIB)
	$(RM) ex6.o

ex7: ex7.o  chkopts
	-$(CLINKER) -o ex7 ex7.o $(PETSC_LIB)
	$(RM) ex7.o

ex8: ex8.o  chkopts
	-$(CLINKER) -o ex8 ex8.o $(PETSC_LIB)
	$(RM) ex8.o

ex9: ex9.o  chkopts
	-$(CLINKER) -o ex9 ex9.o $(PETSC_LIB)
	$(RM) ex9.o

 ex10: ex10.o  chkopts
	-$(CLINKER) -o ex10 ex10.o $(PETSC_LIB)
	$(RM) ex10.o

ex11: ex11.o  chkopts
	-$(CLINKER) -o ex11 ex11.o $(PETSC_LIB)
	$(RM) ex11.o

ex12: ex12.o  chkopts
	-$(FLINKER) -o ex12 ex12.o $(PETSC_FORTRAN_LIB) $(PETSC_LIB)
	$(RM) ex12.o

runex1:
	-@$(MPIRUN) ex1 -test -nox -ts_type euler  -linear_no_matrix
	-@$(MPIRUN) ex1 -test -nox -ts_type euler -nonlinear_no_jacobian 
	-@$(MPIRUN) ex1 -test -nox -ts_type euler -linear_constant_matrix  
	-@$(MPIRUN) ex1 -test -nox -ts_type euler -linear_variable_matrix 
	-@$(MPIRUN) ex1 -test -nox -ts_type euler -nonlinear_jacobian 
	-@$(MPIRUN) ex1 -test -nox -ts_type beuler -linear_no_matrix 
	-@$(MPIRUN) ex1 -test -nox -ts_type beuler -nonlinear_no_jacobian 
	-@$(MPIRUN) ex1 -test -nox -ts_type beuler -linear_constant_matrix  -pc_type lu 
	-@$(MPIRUN) ex1 -test -nox -ts_type beuler -linear_variable_matrix 
	-@$(MPIRUN) ex1 -test -nox -ts_type beuler -nonlinear_jacobian 

runex1f:
	-@$(MPIRUN) ex1f -test -nox -ts_type euler  -linear_no_matrix
	-@$(MPIRUN) ex1f -test -nox -ts_type euler -nonlinear_no_jacobian 
	-@$(MPIRUN) ex1f -test -nox -ts_type euler -linear_constant_matrix  
	-@$(MPIRUN) ex1f -test -nox -ts_type euler -linear_variable_matrix 
	-@$(MPIRUN) ex1f -test -nox -ts_type euler -nonlinear_jacobian 
	-@$(MPIRUN) ex1f -test -nox -ts_type beuler -linear_no_matrix 
	-@$(MPIRUN) ex1f -test -nox -ts_type beuler -nonlinear_no_jacobian 
	-@$(MPIRUN) ex1f -test -nox -ts_type beuler -linear_constant_matrix  -pc_type lu 
	-@$(MPIRUN) ex1f -test -nox -ts_type beuler -linear_variable_matrix 
	-@$(MPIRUN) ex1f -test -nox -ts_type beuler -nonlinear_jacobian 

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
