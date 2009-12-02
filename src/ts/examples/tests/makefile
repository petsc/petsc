
CFLAGS	        = 
FFLAGS	        = 
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/ts/examples/tests/
EXAMPLESC       = ex1.c ex2.c ex3.c ex4.c
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

#----------------------------------------------------------------------------------
NPROCS    = 1  3
runex1_euler:
	-@${MPIEXEC} -n 1 ./ex1 -ts_type euler -linear_constant_matrix -monitor -nox -time 10 > ex1_euler.tmp 2>&1;   \
           if (${DIFF} output/ex1_euler.out ex1_euler.tmp) then true; \
           else echo "Possible problem with ex1_euler, diffs above"; fi; \
           ${RM} -f ex1_euler.tmp

runex1_euler_2:
	-@${MPIEXEC} -n 1 ./ex1 -ts_type euler -linear_variable_matrix -monitor -nox -time 10 > ex1_euler.tmp 2>&1;   \
           if (${DIFF} output/ex1_euler.out ex1_euler.tmp) then true; \
           else echo "Possible problem with ex1_euler_2, diffs above"; fi; \
           ${RM} -f ex1_euler.tmp

runex1_euler_3:
	-@${MPIEXEC} -n 1 ./ex1 -ts_type euler -linear_no_matrix -monitor -nox -time 10 > ex1_euler.tmp 2>&1;   \
           if (${DIFF} output/ex1_euler.out ex1_euler.tmp) then true; \
           else echo "Possible problem with ex1_euler_3, diffs above"; fi; \
           ${RM} -f ex1_euler.tmp

runex1_euler_4:
	-@${MPIEXEC} -n 1 ./ex1 -ts_type euler -nonlinear_jacobian -monitor -nox -time 10 > ex1_euler.tmp 2>&1;   \
           if (${DIFF} output/ex1_euler.out ex1_euler.tmp) then true; \
           else echo "Possible problem with ex1_euler_4, diffs above"; fi; \
           ${RM} -f ex1_euler.tmp

runex1_euler_5:
	-@${MPIEXEC} -n 1 ./ex1 -ts_type euler -nonlinear_no_jacobian -monitor -nox -time 10 > ex1_euler.tmp 2>&1;   \
           if (${DIFF} output/ex1_euler.out ex1_euler.tmp) then true; \
           else echo "Possible problem with ex1_euler_5, diffs above"; fi; \
           ${RM} -f ex1_euler.tmp

runex1_beuler:
	-@touch ex1_beuler.tmp;\
	for np in ${NPROCS}; do \
	  ${MPIEXEC} -n $$np ./ex1 -ts_type beuler -ksp_gmres_cgs_refinement_type refine_always -linear_constant_matrix -monitor -nox -time 10 >> ex1_beuler.tmp 2>&1;   \
	done; \
        if (${DIFF} output/ex1_beuler.out ex1_beuler.tmp) then true; \
        else echo "Possible problem with ex1_beuler, diffs above"; fi; \
        ${RM} -f ex1_beuler.tmp; 
runex1_beuler_2:
	-@touch ex1_beuler.tmp;\
	for np in ${NPROCS}; do \
	  ${MPIEXEC} -n $$np ./ex1 -ts_type beuler -linear_variable_matrix -monitor -nox -time 10 >> ex1_beuler.tmp 2>&1;   \
	done; \
        if (${DIFF} output/ex1_beuler.out ex1_beuler.tmp) then true; \
        else echo "Possible problem with ex1_beuler_2, diffs above"; fi; \
        ${RM} -f ex1_beuler.tmp; 
runex1_beuler_3:
	-@touch ex1_beuler.tmp;\
	for np in ${NPROCS}; do \
	  ${MPIEXEC} -n $$np ./ex1 -ts_type beuler -nonlinear_jacobian -monitor -nox -time 10 >> ex1_beuler.tmp 2>&1;   \
	done; \
        if (${DIFF} output/ex1_beuler.out ex1_beuler.tmp) then true; \
        else echo "Possible problem with ex1_beuler_3, diffs above"; fi; \
        ${RM} -f ex1_beuler.tmp; 
runex1_beuler_4:
	-@touch ex1_beuler.tmp;\
	for np in ${NPROCS}; do \
	  ${MPIEXEC} -n $$np ./ex1 -ts_type beuler -nonlinear_no_jacobian -pc_type none -monitor -nox -time 10 >> ex1_beuler.tmp 2>&1;   \
	done; \
        if (${DIFF} output/ex1_beuler.out ex1_beuler.tmp) then true; \
        else echo "Possible problem with ex1_4, diffs above"; fi; \
        ${RM} -f ex1_beuler.tmp; 

runex1_beuler_5:
	-@touch ex1_beuler.tmp;\
	for np in ${NPROCS}; do \
	   ${MPIEXEC} -n $$np ./ex1 -ts_type beuler -linear_no_matrix -pc_type none -monitor -nox -time 10 >> ex1_beuler.tmp 2>&1;   \
	done; \
        if (${DIFF} output/ex1_beuler.out ex1_beuler.tmp) then true; \
        else echo "Possible problem with ex1_beuler_5, diffs above"; fi; \
        ${RM} -f ex1_beuler.tmp 

runex1_cn:
	-@touch ex1_cn.tmp;\
	for np in ${NPROCS}; do \
	  ${MPIEXEC} -n $$np ./ex1 -ts_type crank-nicholson -linear_constant_matrix -monitor -nox -time 10 >> ex1_cn.tmp 2>&1;   \
        done; \
        if (${DIFF} output/ex1_cn.out ex1_cn.tmp) then true; \
        else echo "Possible problem with ex1_cn, diffs above"; fi; \
        ${RM} -f ex1_cn.tmp 
runex1_cn_2:
	-@touch ex1_cn.tmp;\
	for np in ${NPROCS}; do \
	  ${MPIEXEC} -n $$np ./ex1 -ts_type crank-nicholson -linear_variable_matrix -monitor -nox -time 10 >> ex1_cn.tmp 2>&1;   \
	done; \
        if (${DIFF} output/ex1_cn.out ex1_cn.tmp) then true; \
        else echo "Possible problem with ex1_cn_2, diffs above"; fi; \
        ${RM} -f ex1_cn.tmp
runex1_cn_3:
	-@touch ex1_cn.tmp;\
	for np in ${NPROCS}; do \
	  ${MPIEXEC} -n $$np ./ex1 -ts_type crank-nicholson -linear_no_matrix -pc_type none -monitor -nox -time 10 >> ex1_cn.tmp 2>&1;   \
	done; \
        if (${DIFF} output/ex1_cn.out ex1_cn.tmp) then true; \
        else echo "Possible problem with ex1_cn_3, diffs above"; fi; \
        ${RM} -f ex1_cn.tmp
runex1_cn_4:
	-@touch ex1_cn.tmp;\
	for np in ${NPROCS}; do \
	  ${MPIEXEC} -n $$np ./ex1 -ts_type crank-nicholson -nonlinear_jacobian -monitor -nox -time 10 >> ex1_cn.tmp 2>&1;   \
	done; \
        if (${DIFF} output/ex1_cn.out ex1_cn.tmp) then true; \
        else echo "Possible problem with ex1_cn_4, diffs above"; fi; \
        ${RM} -f ex1_cn.tmp
runex1_cn_5:
	-@touch ex1_cn.tmp;\
	for np in ${NPROCS}; do \
	  ${MPIEXEC} -n $$np ./ex1 -ts_type crank-nicholson -nonlinear_no_jacobian -pc_type none -monitor -nox -time 10 >> ex1_cn.tmp 2>&1;   \
	done; \
        if (${DIFF} output/ex1_cn.out ex1_cn.tmp) then true; \
        else echo "Possible problem with ex1_cn_5, diffs above"; fi; \
        ${RM} -f ex1_cn.tmp; 

runex1f:
	-@${MPIEXEC} -n 1 ./ex1f -test -nox -ts_type euler  -linear_no_matrix
	-@${MPIEXEC} -n 1 ./ex1f -test -nox -ts_type euler -nonlinear_no_jacobian 
	-@${MPIEXEC} -n 1 ./ex1f -test -nox -ts_type euler -linear_constant_matrix  
	-@${MPIEXEC} -n 1 ./ex1f -test -nox -ts_type euler -linear_variable_matrix 
	-@${MPIEXEC} -n 1 ./ex1f -test -nox -ts_type euler -nonlinear_jacobian 
	-@${MPIEXEC} -n 1 ./ex1f -test -nox -ts_type beuler -ksp_gmres_cgs_refinement_type refine_always -linear_no_matrix -pc_type none
	-@${MPIEXEC} -n 1 ./ex1f -test -nox -ts_type beuler -ksp_gmres_cgs_refinement_type refine_always -nonlinear_no_jacobian -pc_type none
	-@${MPIEXEC} -n 1 ./ex1f -test -nox -ts_type beuler -ksp_gmres_cgs_refinement_type refine_always -linear_constant_matrix  -pc_type lu 
	-@${MPIEXEC} -n 1 ./ex1f -test -nox -ts_type beuler -ksp_gmres_cgs_refinement_type refine_always -linear_variable_matrix 
	-@${MPIEXEC} -n 1 ./ex1f -test -nox -ts_type beuler -ksp_gmres_cgs_refinement_type refine_always -nonlinear_jacobian 

runex3:
	-@${MPIEXEC} -n 1 ./ex3 > ex3.tmp 2>&1;	  \
	   if (${DIFF} output/ex3.out ex3.tmp) then true; \
	   else echo "Possible problem with ex3, diffs above"; fi; \
	   ${RM} -f ex3.tmp

runex3_2:
	-@${MPIEXEC} -n 1 ./ex3 -useAlhs > ex3.tmp 2>&1;	  \
	   if (${DIFF} output/ex3.out ex3.tmp) then true; \
	   else echo "Possible problem with ex3_2, diffs above"; fi; \
	   ${RM} -f ex3.tmp

runex4:
	-@${MPIEXEC} -n 1 ./ex4 -ts_fd -ts_type beuler> ex4.tmp 2>&1;	  \
	   if (${DIFF} output/ex4.out ex4.tmp) then true; \
	   else echo "Possible problem with ex4, diffs above"; fi; \
	   ${RM} -f ex4.tmp
runex4_1:
	-@${MPIEXEC} -n 2 ./ex4 -ts_fd -ts_type beuler> ex4.tmp 2>&1;	  \
	   if (${DIFF} output/ex4.out ex4.tmp) then true; \
	   else echo "Possible problem with ex4_1, diffs above"; fi; \
	   ${RM} -f ex4.tmp
runex4_2:
	-@${MPIEXEC} -n 2 ./ex4 -ts_fd -ts_type beuler -snes_mf_operator> ex4.tmp 2>&1;	  \
	   if (${DIFF} output/ex4.out ex4.tmp) then true; \
	   else echo "Possible problem with ex4_2, diffs above"; fi; \
	   ${RM} -f ex4.tmp
runex4_3:
	-@${MPIEXEC} -n 2 ./ex4 -ts_fd -ts_type crank-nicholson > ex4.tmp 2>&1;	  \
	   if (${DIFF} output/ex4_3.out ex4.tmp) then true; \
	   else echo "Possible problem with ex4_3, diffs above"; fi; \
	   ${RM} -f ex4.tmp
runex4_4:
	-@${MPIEXEC} -n 2 ./ex4 -ts_fd -ts_type crank-nicholson -snes_mf_operator > ex4.tmp 2>&1;	  \
	   if (${DIFF} output/ex4_3.out ex4.tmp) then true; \
	   else echo "Possible problem with ex4_4, diffs above"; fi; \
	   ${RM} -f ex4.tmp
runex4_5:
	-@${MPIEXEC} -n 1 ./ex4 -ts_type beuler -ts_fd -fd_color -mat_coloring_type sl > ex4.tmp 2>&1;	  \
	   if (${DIFF} output/ex4.out ex4.tmp) then true; \
	   else echo "Possible problem with ex4_5, diffs above"; fi; \
	   ${RM} -f ex4.tmp
runex4_6:
	-@${MPIEXEC} -n 2 ./ex4 -ts_type beuler -ts_fd -fd_color -mat_coloring_type sl > ex4.tmp 2>&1;	  \
	   if (${DIFF} output/ex4.out ex4.tmp) then true; \
	   else echo "Possible problem with ex4_6, diffs above"; fi; \
	   ${RM} -f ex4.tmp

runex4_7:
	-@${MPIEXEC} -n 1 ./ex4 -ts_fd -ts_type beuler -test_PostStep > ex4.tmp 2>&1;	  \
	   if (${DIFF} output/ex4_7.out ex4.tmp) then true; \
	   else echo "Possible problem with ex4_7, diffs above"; fi; \
	   ${RM} -f ex4.tmp

TESTEXAMPLES_C		  = ex1.PETSc runex1_euler runex1_euler_2 runex1_euler_3 runex1_euler_4 runex1_euler_5 \
                            runex1_beuler runex1_beuler_2 runex1_beuler_3 runex1_beuler_4 runex1_beuler_5 \
                            runex1_cn runex1_cn_2 runex1_cn_3 ex1.rm \
                            ex4.PETSc runex4 runex4_1 runex4_2 runex4_3 runex4_4 runex4_5 runex4_6 \
                            runex4_7 ex4.rm

testexamples_C_NoComplex  = ex3.PETSc runex3 runex3_2 ex3.rm
TESTEXAMPLES_C_X11	  = 
TESTEXAMPLES_FORTRAN	  = ex1f.PETSc runex1f ex1f.rm
TESTEXAMPLES_C_X11_MPIUNI = 
TESTEXAMPLES_13		  = ex2.PETSc ex2.rm ex4.PETSc ex4.rm 

include ${PETSC_DIR}/conf/test
