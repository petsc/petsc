
CFLAGS	        = 
FFLAGS	        = 
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/ts/examples/tests/
EXAMPLESC       = ex1.c ex2.c ex3.c ex4.c ex5.c
EXAMPLESF       = ex1f.F
EXAMPLESFH      = ex1f.h
MANSEC          = TS

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

ex1: ex1.o  chkopts
	-${CLINKER} -o ex1 ex1.o  ${PETSC_TS_LIB}
	${RM} ex1.o

ex1f: ex1f.o  chkopts
	-${FLINKER} -o ex1f ex1f.o  ${PETSC_TS_LIB}
	${RM} ex1f.o

ex2: ex2.o  chkopts
	-${CLINKER} -o ex2 ex2.o ${PETSC_TS_LIB}
	${RM} ex2.o

ex3: ex3.o  chkopts
	-${CLINKER} -o ex3 ex3.o ${PETSC_TS_LIB}
	${RM} ex3.o

ex4: ex4.o  chkopts
	-${CLINKER} -o ex4 ex4.o ${PETSC_TS_LIB}
	${RM} ex4.o

ex5: ex5.o  chkopts
	-${CLINKER} -o ex5 ex5.o ${PETSC_TS_LIB}
	${RM} ex5.o

#----------------------------------------------------------------------------------
NPROCS    = 1  3
runex1_euler:
	-@${MPIEXEC} -n 1 ./ex1 -ts_type euler -linear_constant_matrix -monitor -nox -time 10 > ex1_euler.tmp 2>&1;   \
           ${DIFF} output/ex1_euler.out ex1_euler.tmp || echo  ${PWD} "\nPossible problem with ex1_euler, diffs above \n========================================="; \
           ${RM} -f ex1_euler.tmp

runex1_euler_2:
	-@${MPIEXEC} -n 1 ./ex1 -ts_type euler -linear_variable_matrix -monitor -nox -time 10 > ex1_euler.tmp 2>&1;   \
           ${DIFF} output/ex1_euler.out ex1_euler.tmp || echo  ${PWD} "\nPossible problem with ex1_euler_2, diffs above \n========================================="; \
           ${RM} -f ex1_euler.tmp

runex1_euler_3:
	-@${MPIEXEC} -n 1 ./ex1 -ts_type euler -linear_no_matrix -monitor -nox -time 10 > ex1_euler.tmp 2>&1;   \
           ${DIFF} output/ex1_euler.out ex1_euler.tmp || echo  ${PWD} "\nPossible problem with ex1_euler_3, diffs above \n========================================="; \
           ${RM} -f ex1_euler.tmp

runex1_euler_4:
	-@${MPIEXEC} -n 1 ./ex1 -ts_type euler -nonlinear_jacobian -monitor -nox -time 10 > ex1_euler.tmp 2>&1;   \
           ${DIFF} output/ex1_euler.out ex1_euler.tmp || echo  ${PWD} "\nPossible problem with ex1_euler_4, diffs above \n========================================="; \
           ${RM} -f ex1_euler.tmp

runex1_euler_5:
	-@${MPIEXEC} -n 1 ./ex1 -ts_type euler -nonlinear_no_jacobian -monitor -nox -time 10 > ex1_euler.tmp 2>&1;   \
           ${DIFF} output/ex1_euler.out ex1_euler.tmp || echo  ${PWD} "\nPossible problem with ex1_euler_5, diffs above \n========================================="; \
           ${RM} -f ex1_euler.tmp

runex1_beuler:
	-@touch ex1_beuler.tmp;\
	for np in ${NPROCS}; do \
	  ${MPIEXEC} -n $$np ./ex1 -ts_type beuler -ksp_gmres_cgs_refinement_type refine_always -linear_constant_matrix -monitor -nox -time 10 >> ex1_beuler.tmp 2>&1;   \
	done; \
        ${DIFF} output/ex1_beuler.out ex1_beuler.tmp || echo  ${PWD} "\nPossible problem with ex1_beuler, diffs above \n========================================="; \
        ${RM} -f ex1_beuler.tmp;
runex1_beuler_2:
	-@touch ex1_beuler.tmp;\
	for np in ${NPROCS}; do \
	  ${MPIEXEC} -n $$np ./ex1 -ts_type beuler -linear_variable_matrix -monitor -nox -time 10 >> ex1_beuler.tmp 2>&1;   \
	done; \
        ${DIFF} output/ex1_beuler.out ex1_beuler.tmp || echo  ${PWD} "\nPossible problem with ex1_beuler_2, diffs above \n========================================="; \
        ${RM} -f ex1_beuler.tmp;
runex1_beuler_3:
	-@touch ex1_beuler.tmp;\
	for np in ${NPROCS}; do \
	  ${MPIEXEC} -n $$np ./ex1 -ts_type beuler -nonlinear_jacobian -monitor -nox -time 10 >> ex1_beuler.tmp 2>&1;   \
	done; \
        ${DIFF} output/ex1_beuler.out ex1_beuler.tmp || echo  ${PWD} "\nPossible problem with ex1_beuler_3, diffs above \n========================================="; \
        ${RM} -f ex1_beuler.tmp;
runex1_beuler_4:
	-@touch ex1_beuler.tmp;\
	for np in ${NPROCS}; do \
	  ${MPIEXEC} -n $$np ./ex1 -ts_type beuler -nonlinear_no_jacobian -pc_type none -monitor -nox -time 10 >> ex1_beuler.tmp 2>&1;   \
	done; \
        ${DIFF} output/ex1_beuler.out ex1_beuler.tmp || echo  ${PWD} "\nPossible problem with ex1_4, diffs above \n========================================="; \
        ${RM} -f ex1_beuler.tmp; 

runex1_beuler_5:
	-@touch ex1_beuler.tmp;\
	for np in ${NPROCS}; do \
	   ${MPIEXEC} -n $$np ./ex1 -ts_type beuler -linear_no_matrix -pc_type none -monitor -nox -time 10 >> ex1_beuler.tmp 2>&1;   \
	done; \
        ${DIFF} output/ex1_beuler.out ex1_beuler.tmp || echo  ${PWD} "\nPossible problem with ex1_beuler_5, diffs above \n========================================="; \
        ${RM} -f ex1_beuler.tmp 

runex1_cn:
	-@touch ex1_cn.tmp;\
	for np in ${NPROCS}; do \
	  ${MPIEXEC} -n $$np ./ex1 -ts_type cn -linear_constant_matrix -monitor -nox -time 10 >> ex1_cn.tmp 2>&1;   \
        done; \
        ${DIFF} output/ex1_cn.out ex1_cn.tmp || echo  ${PWD} "\nPossible problem with ex1_cn, diffs above \n========================================="; \
        ${RM} -f ex1_cn.tmp 
runex1_cn_2:
	-@touch ex1_cn.tmp;\
	for np in ${NPROCS}; do \
	  ${MPIEXEC} -n $$np ./ex1 -ts_type cn -linear_variable_matrix -monitor -nox -time 10 >> ex1_cn.tmp 2>&1;   \
	done; \
        ${DIFF} output/ex1_cn.out ex1_cn.tmp || echo  ${PWD} "\nPossible problem with ex1_cn_2, diffs above \n========================================="; \
        ${RM} -f ex1_cn.tmp
runex1_cn_3:
	-@touch ex1_cn.tmp;\
	for np in ${NPROCS}; do \
	  ${MPIEXEC} -n $$np ./ex1 -ts_type cn -linear_no_matrix -pc_type none -monitor -nox -time 10 >> ex1_cn.tmp 2>&1;   \
	done; \
        ${DIFF} output/ex1_cn.out ex1_cn.tmp || echo  ${PWD} "\nPossible problem with ex1_cn_3, diffs above \n========================================="; \
        ${RM} -f ex1_cn.tmp
runex1_cn_4:
	-@touch ex1_cn.tmp;\
	for np in ${NPROCS}; do \
	  ${MPIEXEC} -n $$np ./ex1 -ts_type cn -nonlinear_jacobian -monitor -nox -time 10 >> ex1_cn.tmp 2>&1;   \
	done; \
        ${DIFF} output/ex1_cn.out ex1_cn.tmp || echo  ${PWD} "\nPossible problem with ex1_cn_4, diffs above \n========================================="; \
        ${RM} -f ex1_cn.tmp
runex1_cn_5:
	-@touch ex1_cn.tmp;\
	for np in ${NPROCS}; do \
	  ${MPIEXEC} -n $$np ./ex1 -ts_type cn -nonlinear_no_jacobian -pc_type none -monitor -nox -time 10 >> ex1_cn.tmp 2>&1;   \
	done; \
        ${DIFF} output/ex1_cn.out ex1_cn.tmp || echo  ${PWD} "\nPossible problem with ex1_cn_5, diffs above \n========================================="; \
        ${RM} -f ex1_cn.tmp; 

runex1f:
	-@${MPIEXEC} -n 1 ./ex1f -test -nox -ts_type euler  -linear_no_matrix
	-@${MPIEXEC} -n 1 ./ex1f -test -nox -ts_type euler -nonlinear_no_jacobian 
	-@${MPIEXEC} -n 1 ./ex1f -test -nox -ts_type euler -linear_constant_matrix  
	-@${MPIEXEC} -n 1 ./ex1f -test -nox -ts_type euler -nonlinear_jacobian 
	-@${MPIEXEC} -n 1 ./ex1f -test -nox -ts_type beuler -ksp_gmres_cgs_refinement_type refine_always -linear_no_matrix -pc_type none
	-@${MPIEXEC} -n 1 ./ex1f -test -nox -ts_type beuler -ksp_gmres_cgs_refinement_type refine_always -nonlinear_no_jacobian -pc_type none
	-@${MPIEXEC} -n 1 ./ex1f -test -nox -ts_type beuler -ksp_gmres_cgs_refinement_type refine_always -linear_constant_matrix  -pc_type lu 
	-@${MPIEXEC} -n 1 ./ex1f -test -nox -ts_type beuler -ksp_gmres_cgs_refinement_type refine_always -nonlinear_jacobian 

runex2_euler:
	-@${MPIEXEC} -n 1 ./ex2 -ts_type euler > ex2_euler.tmp 2>&1;   \
           ${DIFF} output/ex2_euler.out ex2_euler.tmp || echo  ${PWD} "\nPossible problem with ex2_euler, diffs above \n========================================="; \
           ${RM} -f ex2_euler.tmp

runex2_beuler:
	-@${MPIEXEC} -n 1 ./ex2 -ts_type beuler > ex2_beuler.tmp 2>&1;   \
           ${DIFF} output/ex2_beuler.out ex2_beuler.tmp || echo  ${PWD} "\nPossible problem with ex2_beuler, diffs above \n========================================="; \
           ${RM} -f ex2_beuler.tmp

runex2_sundials:
	-@${MPIEXEC} -n 1 ./ex2 -ts_type sundials -ts_sundials_monitor_steps false > ex2_sundials.tmp 2>&1;   \
           ${DIFF} output/ex2_sundials.out ex2_sundials.tmp || echo  ${PWD} "\nPossible problem with ex2_sundials, diffs above \n========================================="; \
           ${RM} -f ex2_sundials.tmp

runex3:
	-@${MPIEXEC} -n 1 ./ex3 > ex3.tmp 2>&1;	  \
	   ${DIFF} output/ex3.out ex3.tmp || echo  ${PWD} "\nPossible problem with ex3, diffs above \n========================================="; \
	   ${RM} -f ex3.tmp

runex3_2:
	-@${MPIEXEC} -n 1 ./ex3 -useAlhs > ex3.tmp 2>&1;	  \
	   ${DIFF} output/ex3.out ex3.tmp || echo  ${PWD} "\nPossible problem with ex3_2, diffs above \n========================================="; \
	   ${RM} -f ex3.tmp

runex4:
	-@${MPIEXEC} -n 1 ./ex4 -ts_fd -ts_type beuler> ex4.tmp 2>&1;	  \
	   ${DIFF} output/ex4.out ex4.tmp || echo  ${PWD} "\nPossible problem with ex4, diffs above \n========================================="; \
	   ${RM} -f ex4.tmp
runex4_2:
	-@${MPIEXEC} -n 2 ./ex4 -ts_fd -ts_type beuler > ex4.tmp 2>&1;	  \
	   ${DIFF} output/ex4.out ex4.tmp || echo  ${PWD} "\nPossible problem with ex4_2, diffs above \n========================================="; \
	   ${RM} -f ex4.tmp
runex4_3:
	-@${MPIEXEC} -n 1 ./ex4 -ts_fd -ts_type cn > ex4.tmp 2>&1;	  \
	   ${DIFF} output/ex4_3.out ex4.tmp || echo  ${PWD} "\nPossible problem with ex4_3, diffs above \n========================================="; \
	   ${RM} -f ex4.tmp
runex4_4:
	-@${MPIEXEC} -n 2 ./ex4 -ts_fd -ts_type cn > ex4.tmp 2>&1;	  \
	   ${DIFF} output/ex4_3.out ex4.tmp || echo  ${PWD} "\nPossible problem with ex4_4, diffs above \n========================================="; \
	   ${RM} -f ex4.tmp
runex4_5:
	-@${MPIEXEC} -n 1 ./ex4 -ts_type beuler -ts_fd -fd_color -mat_coloring_type sl > ex4.tmp 2>&1;	  \
	   ${DIFF} output/ex4.out ex4.tmp || echo  ${PWD} "\nPossible problem with ex4_5, diffs above \n========================================="; \
	   ${RM} -f ex4.tmp
runex4_6:
	-@${MPIEXEC} -n 2 ./ex4 -ts_type beuler -ts_fd -fd_color -mat_coloring_type sl > ex4.tmp 2>&1;	  \
	   ${DIFF} output/ex4.out ex4.tmp || echo  ${PWD} "\nPossible problem with ex4_6, diffs above \n========================================="; \
	   ${RM} -f ex4.tmp

runex4_7:
	-@${MPIEXEC} -n 1 ./ex4 -ts_fd -ts_type beuler -test_PostStep > ex4.tmp 2>&1;	  \
	   ${DIFF} output/ex4_7.out ex4.tmp || echo  ${PWD} "\nPossible problem with ex4_7, diffs above \n========================================="; \
	   ${RM} -f ex4.tmp

runex5:
	-@${MPIEXEC} -n 1 ./ex5 -ts_max_steps 130 > ex5.tmp 2>&1;	  \
	   ${DIFF} output/ex5.out ex5.tmp || echo  ${PWD} "\nPossible problem with ex5_1, diffs above \n========================================="; \
	   ${RM} -f ex5.tmp

runex5_2:
	-@${MPIEXEC} -n 4 ./ex5 -ts_max_steps 130 > ex5.tmp 2>&1;	  \
	   ${DIFF} output/ex5.out ex5.tmp || echo  ${PWD} "\nPossible problem with ex5_2, diffs above \n========================================="; \
	   ${RM} -f ex5.tmp

TESTEXAMPLES_C		  = ex1.PETSc runex1_euler_4 runex1_euler_5 \
                            runex1_beuler_3 runex1_beuler_4 ex1.rm \
                            ex4.PETSc runex4 runex4_2 runex4_3 runex4_4 runex4_5 runex4_6 \
                            runex4_7 ex4.rm 

testexamples_C_NoComplex  = ex3.PETSc runex3 runex3_2 ex3.rm ex5.PETSc runex5 runex5_2 ex5.rm
TESTEXAMPLES_C_X	  = 
TESTEXAMPLES_FORTRAN	  = ex1f.PETSc ex1f.rm
TESTEXAMPLES_C_X_MPIUNI = 
TESTEXAMPLES_13		  = ex2.PETSc ex2.rm ex4.PETSc ex4.rm 

include ${PETSC_DIR}/conf/test
