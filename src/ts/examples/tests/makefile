
CFLAGS	        = 
FFLAGS	        = 
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/ts/examples/tests/
EXAMPLESC       = ex1.c ex2.c ex4.c
EXAMPLESF       = ex1f.F
EXAMPLESFH      = ex1f.h
MANSEC          = TS

include ${PETSC_DIR}/bmake/common/base

ex1: ex1.o  chkopts
	-${CLINKER} -o ex1 ex1.o  ${PETSC_TS_LIB}
	${RM} ex1.o

ex1f: ex1f.o  chkopts
	-${FLINKER} -o ex1f ex1f.o  ${PETSC_TS_LIB}
	${RM} ex1f.o

ex2: ex2.o  chkopts
	-${CLINKER} -o ex2 ex2.o -L${PETSC_LIB_DIR} -lpetscts ${PETSC_TS_LIB}
	${RM} ex2.o

ex3: ex3.o  chkopts
	-${CLINKER} -o ex3 ex3.o -L${PETSC_LIB_DIR} -lpetscts 
	${RM} ex3.o

ex4: ex4.o  chkopts
	-${CLINKER} -o ex4 ex4.o -L${PETSC_LIB_DIR} -lpetscts ${PETSC_TS_LIB}
	${RM} ex4.o

ex5: ex5.o  chkopts
	-${CLINKER} -o ex5 ex5.o -L${PETSC_LIB_DIR} -lpetscts ${PETSC_TS_LIB}
	${RM} ex5.o

ex6: ex6.o  chkopts
	-${CLINKER} -o ex6 ex6.o ${PETSC_TS_LIB}
	${RM} ex6.o

ex7: ex7.o  chkopts
	-${CLINKER} -o ex7 ex7.o ${PETSC_TS_LIB}
	${RM} ex7.o

ex8: ex8.o  chkopts
	-${CLINKER} -o ex8 ex8.o ${PETSC_TS_LIB}
	${RM} ex8.o

ex9: ex9.o  chkopts
	-${CLINKER} -o ex9 ex9.o ${PETSC_TS_LIB}
	${RM} ex9.o

ex10: ex10.o  chkopts
	-${CLINKER} -o ex10 ex10.o ${PETSC_TS_LIB}
	${RM} ex10.o

ex11: ex11.o  chkopts
	-${CLINKER} -o ex11 ex11.o ${PETSC_TS_LIB}
	${RM} ex11.o

ex12: ex12.o  chkopts
	-${FLINKER} -o ex12 ex12.o ${PETSC_TS_LIB}
	${RM} ex12.o
#----------------------------------------------------------------------------------
runex1:
	-@${MPIRUN} -np 1 ./ex1 -test -nox -ts_type euler  -linear_no_matrix > ex1_1.tmp 2>&1; 
	-@${MPIRUN} -np 1 ./ex1 -test -nox -ts_type euler -nonlinear_no_jacobian >> ex1_1.tmp 2>&1; 
	-@${MPIRUN} -np 1 ./ex1 -test -nox -ts_type euler -linear_constant_matrix >> ex1_1.tmp 2>&1; 
	-@${MPIRUN} -np 1 ./ex1 -test -nox -ts_type euler -linear_variable_matrix >> ex1_1.tmp 2>&1; 
	-@${MPIRUN} -np 1 ./ex1 -test -nox -ts_type euler  -nonlinear_jacobian >> ex1_1.tmp 2>&1; 
	-@${MPIRUN} -np 1 ./ex1 -test -nox -ts_type beuler -ksp_gmres_cgs_refinement_type refine_always -linear_no_matrix -pc_type none  -ksp_rtol 1.e-8 >> ex1_1.tmp 2>&1; 
	-@${MPIRUN} -np 1 ./ex1 -test -nox -ts_type beuler -ksp_gmres_cgs_refinement_type refine_always -nonlinear_no_jacobian -pc_type none >> ex1_1.tmp 2>&1; 
	-@${MPIRUN} -np 1 ./ex1 -test -nox -ts_type beuler -ksp_gmres_cgs_refinement_type refine_always -linear_constant_matrix  -pc_type lu >> ex1_1.tmp 2>&1; 
	-@${MPIRUN} -np 1 ./ex1 -test -nox -ts_type beuler -ksp_gmres_cgs_refinement_type refine_always -linear_variable_matrix >> ex1_1.tmp 2>&1; 
	-@${MPIRUN} -np 1 ./ex1 -test -nox -ts_type beuler -ksp_gmres_cgs_refinement_type refine_always -nonlinear_jacobian >> ex1_1.tmp 2>&1; \
         if (${DIFF} output/ex1_1.out ex1_1.tmp) then true; \
           else echo "Possible problem with ex1_1, diffs above"; fi; \
           ${RM} -f ex1_1.tmp

runex1_euler:
	-@${MPIRUN} -np 1  ./ex1 -test -ts_type euler -linear_constant_matrix -monitor -nox> ex1_euler.tmp 2>&1;   \
           if (${DIFF} output/ex1_euler.out ex1_euler.tmp) then true; \
           else echo "Possible problem with ex1_euler, diffs above"; fi; \
           ${RM} -f ex1_euler.tmp

runex1_beuler:
	-@${MPIRUN} -np 1 ./ex1 -test -ts_type beuler -linear_constant_matrix -pc_type lu -monitor -nox> ex1_beuler.tmp 2>&1;   \
           if (${DIFF} output/ex1_beuler.out ex1_beuler.tmp) then true; \
           else echo "Possible problem with ex1_beuler, diffs above"; fi; \
           ${RM} -f ex1_beuler.tmp

runex1_cn:
	-@${MPIRUN} -np 1 ./ex1 -test -ts_type crank-nicholson -linear_constant_matrix -monitor -nox> ex1_cn.tmp 2>&1;   \
           if (${DIFF} output/ex1_cn.out ex1_cn.tmp) then true; \
           else echo "Possible problem with ex1_cn, diffs above"; fi; \
           ${RM} -f ex1_cn.tmp

runex1f:
	-@${MPIRUN} -np 1 ./ex1f -test -nox -ts_type euler  -linear_no_matrix
	-@${MPIRUN} -np 1 ./ex1f -test -nox -ts_type euler -nonlinear_no_jacobian 
	-@${MPIRUN} -np 1 ./ex1f -test -nox -ts_type euler -linear_constant_matrix  
	-@${MPIRUN} -np 1 ./ex1f -test -nox -ts_type euler -linear_variable_matrix 
	-@${MPIRUN} -np 1 ./ex1f -test -nox -ts_type euler -nonlinear_jacobian 
	-@${MPIRUN} -np 1 ./ex1f -test -nox -ts_type beuler -ksp_gmres_cgs_refinement_type refine_always -linear_no_matrix -pc_type none
	-@${MPIRUN} -np 1 ./ex1f -test -nox -ts_type beuler -ksp_gmres_cgs_refinement_type refine_always -nonlinear_no_jacobian -pc_type none
	-@${MPIRUN} -np 1 ./ex1f -test -nox -ts_type beuler -ksp_gmres_cgs_refinement_type refine_always -linear_constant_matrix  -pc_type lu 
	-@${MPIRUN} -np 1 ./ex1f -test -nox -ts_type beuler -ksp_gmres_cgs_refinement_type refine_always -linear_variable_matrix 
	-@${MPIRUN} -np 1 ./ex1f -test -nox -ts_type beuler -ksp_gmres_cgs_refinement_type refine_always -nonlinear_jacobian 

TESTEXAMPLES_C		  = ex1.PETSc runex1 runex1_euler runex1_beuler runex1_cn ex1.rm 
TESTEXAMPLES_C_X11	  = 
TESTEXAMPLES_FORTRAN	  = ex1f.PETSc runex1f ex1f.rm
TESTEXAMPLES_C_X11_MPIUNI = 
TESTEXAMPLES_13		  = ex2.PETSc ex2.rm ex4.PETSc ex4.rm 

include ${PETSC_DIR}/bmake/common/test
