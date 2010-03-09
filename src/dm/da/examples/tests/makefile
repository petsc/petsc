
CFLAGS	        = 
FFLAGS	        = 
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/dm/da/examples/tests/
EXAMPLESC       = ex1.c ex2.c ex3.c ex4.c ex5.c ex6.c ex7.c ex8.c ex9.c ex10.c\
                  ex11.c ex12.c ex12.m ex13.c ex14.c ex15.c ex16.c ex17.c ex18.c ex19.c \
	          ex20.c ex22.c ex23.c ex24.c ex25.c ex26.c ex27.c ex28.c ex29.c ex30.c \
	          ex31.c ex32.c
EXAMPLESF       = ex21f.F
MANSEC          = DA

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

ex1: ex1.o   chkopts
	-${CLINKER} -o ex1 ex1.o  ${PETSC_DM_LIB}
	${RM} -f ex1.o

ex2: ex2.o  chkopts 
	-${CLINKER} -o ex2 ex2.o  ${PETSC_DM_LIB}
	${RM} -f ex2.o

ex3: ex3.o  chkopts 
	-${CLINKER} -o ex3 ex3.o  ${PETSC_DM_LIB}
	${RM} -f ex3.o

ex4: ex4.o   chkopts
	-${CLINKER} -o ex4 ex4.o  ${PETSC_DM_LIB}
	${RM} -f ex4.o

ex5: ex5.o  chkopts 
	-${CLINKER} -o ex5 ex5.o  ${PETSC_DM_LIB}
	${RM} -f ex5.o

ex6: ex6.o   chkopts
	-${CLINKER} -o ex6 ex6.o  ${PETSC_DM_LIB}
	${RM} -f ex6.o

ex7: ex7.o   chkopts
	-${CLINKER} -o ex7 ex7.o  ${PETSC_DM_LIB}
	${RM} -f ex7.o

ex8: ex8.o   chkopts
	-${CLINKER} -o ex8 ex8.o  ${PETSC_DM_LIB}
	${RM} -f ex8.o

ex9: ex9.o   chkopts
	-${CLINKER} -o ex9 ex9.o  ${PETSC_DM_LIB}
	${RM} -f ex9.o

ex10: ex10.o   chkopts
	-${CLINKER} -o ex10 ex10.o  ${PETSC_DM_LIB}
	${RM} -f ex10.o

ex11: ex11.o   chkopts
	-${CLINKER} -o ex11 ex11.o  ${PETSC_DM_LIB}
	${RM} -f ex11.o

ex12: ex12.o   chkopts
	-${CLINKER} -o ex12 ex12.o  ${PETSC_DM_LIB}
	${RM} -f ex12.o

ex13: ex13.o   chkopts
	-${CLINKER} -o ex13 ex13.o  ${PETSC_DM_LIB}
	${RM} -f ex13.o

ex14: ex14.o   chkopts
	-${CLINKER} -o ex14 ex14.o  ${PETSC_DM_LIB}
	${RM} -f ex14.o

ex15: ex15.o   chkopts
	-${CLINKER} -o ex15 ex15.o  ${PETSC_DM_LIB}
	${RM} -f ex15.o

ex16: ex16.o   chkopts
	-${CLINKER} -o ex16 ex16.o  ${PETSC_DM_LIB}
	${RM} -f ex16.o

ex17: ex17.o   chkopts
	-${CLINKER} -o ex17 ex17.o  ${PETSC_DM_LIB}
	${RM} -f ex17.o

ex18: ex18.o   chkopts
	-${CLINKER} -o ex18 ex18.o  ${PETSC_DM_LIB}
	${RM} -f ex18.o

ex19: ex19.o   chkopts
	-${CLINKER} -o ex19 ex19.o  ${PETSC_DM_LIB}
	${RM} -f ex19.o

ex20: ex20.o   chkopts
	-${CLINKER} -o ex20 ex20.o  ${PETSC_DM_LIB}
	${RM} -f ex20.o

ex21f: ex21f.o  chkopts
	-${FLINKER} -o ex21f ex21f.o  ${PETSC_LIB}
	${RM} -f ex21f.o

ex22: ex22.o   chkopts
	-${CLINKER} -o ex22 ex22.o  ${PETSC_DM_LIB}
	${RM} -f ex22.o

ex23: ex23.o   chkopts
	-${CLINKER} -o ex23 ex23.o  ${PETSC_DM_LIB}
	${RM} -f ex23.o

ex24: ex24.o   chkopts
	-${CLINKER} -o ex24 ex24.o  ${PETSC_DM_LIB}
	${RM} -f ex24.o

ex25: ex25.o   chkopts
	-${CLINKER} -o ex25 ex25.o  ${PETSC_DM_LIB}
	${RM} -f ex25.o

ex26: ex26.o   chkopts
	-${CLINKER} -o ex26 ex26.o  ${PETSC_DM_LIB}
	${RM} -f ex26.o

ex27: ex27.o   chkopts
	-${CLINKER} -o ex27 ex27.o  ${PETSC_DM_LIB}
	${RM} -f ex27.o

ex28: ex28.o   chkopts
	-${CLINKER} -o ex28 ex28.o  ${PETSC_DM_LIB}
	${RM} -f ex28.o

ex29: ex29.o   chkopts
	-${CLINKER} -o ex29 ex29.o  ${PETSC_DM_LIB}
	${RM} -f ex29.o

ex30: ex30.o   chkopts
	-${CLINKER} -o ex30 ex30.o  ${PETSC_DM_LIB}
	${RM} -f ex30.o

ex31: ex31.o   chkopts
	-${CLINKER} -o ex31 ex31.o  ${PETSC_DM_LIB}
	${RM} -f ex31.o

ex32: ex32.o   chkopts
	-${CLINKER} -o ex32 ex32.o  ${PETSC_DM_LIB}
	${RM} -f ex32.o

#-------------------------------------------------------------------------------
runex1:
	-@${MPIEXEC} -n 2 ./ex1 -nox > ex1_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex1_1.out ex1_1.tmp) then true; \
	   else echo "Possible problem with ex1_1, diffs above"; fi; \
	   ${RM} -f ex1_1.tmp
runex2:
	-@${MPIEXEC} -n 2 ./ex2 > ex2_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex2_1.out ex2_1.tmp) then true; \
	   else echo "Possible problem with ex2_1, diffs above"; fi; \
	   ${RM} -f ex2_1.tmp
runex2_2:
	-@${MPIEXEC} -n 3 ./ex2 -wrap nonperiodic -local_print > ex2_2.tmp 2>&1;	  \
	   if (${DIFF} output/ex2_2.out ex2_2.tmp) then true; \
	   else echo "Possible problem with ex2_2, diffs above"; fi; \
	   ${RM} -f ex2_2.tmp
runex2_3:
	-@${MPIEXEC} -n 3 ./ex2 -wrap xyzghosted -local_print > ex2_3.tmp 2>&1;	  \
	   if (${DIFF} output/ex2_3.out ex2_3.tmp) then true; \
	   else echo "Possible problem with ex2_3, diffs above"; fi; \
	   ${RM} -f ex2_3.tmp
runex3:
	-@${MPIEXEC} -n 3 ./ex3 -time 50
runex4:
	-@${MPIEXEC} -n 4 ./ex4  -nox > ex4_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex4_1.out ex4_1.tmp) then true; \
	   else echo "Possible problem with ex4_1, diffs above"; fi; \
	   ${RM} -f ex4_1.tmp
runex5:
	-@${MPIEXEC} -n 3 ./ex5 -time 50
runex6:
	-@${MPIEXEC} -n 6 ./ex6 -test_order
runex12:
	-@echo "path([getenv('PETSC_DIR') '/bin/matlab'],path); ex12(2) ; exit" | ${MATLAB_COMMAND} -display ${DISPLAY} > ex12_1.tmp 2>&1; grep -v flops ex12_1.tmp | grep -v Completed | grep -v About | grep -v Color | grep -v "at line 27" > ex12_1.tmp2 ; if (${DIFF} output/ex12_1.out ex12_1.tmp2) then true; \
	   else echo "Possible problem with PETSC SOCKET MATLAB INTERFACE ex12_1, diffs above"; fi; \
	   ${RM} -f ex12_1.tmp ex12_1.tmp2

# ex13 uses the output of ex14
runex13:
	-@${MPIEXEC} -n 3 ./ex13
	-@${RM} -f daoutput daoutput.info 
runex14:
	-@${MPIEXEC} -n 4 ./ex14 ; sleep 1

runex20:
	-@${MPIEXEC} -n 2 ./ex20 -2d -periodic 1 -dof 2 -stencil_width 1 -stencil_type 0

runex20_2:
	-@for d in -1d -2d -3d; do \
            for np in 1 2 3 4; do \
              for per in 0 1; do \
                for dof in 1 2 3; do \
                  for stencil_width in 1 2; do \
                    for stencil_type in 0 1; do \
                      ${MPIEXEC} -n $$np ./ex20 $$d -periodic $$per -dof $$dof -stencil_width $$stencil_width -stencil_type $$stencil_type ;\
                    done; \
                  done; \
                done; \
              done; \
            done; \
          done;

runex23:
	-@for d in -1d -2d -3d; do \
            for np in 1 2 3 4; do \
              for dof in 1 2 3; do \
                for stencil_type in 0 1; do \
                  ${MPIEXEC} -n $$np ./ex23 $$d -dof $$dof -stencil_type $$stencil_type ;\
                done; \
              done; \
            done; \
          done;

runex24:
	-@${MPIEXEC} -n 1 ./ex24  | grep -v Process > ex24_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex24_1.out ex24_1.tmp) then true; \
	   else echo "Possible problem with ex24_1 1 process, diffs above"; fi; ${RM} -f ex24_1.tmp
	-@${MPIEXEC} -n 2 ./ex24  | grep -v Process > ex24_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex24_1.out ex24_1.tmp) then true; \
	   else echo "Possible problem with ex24_1 2 processes, diffs above"; fi; ${RM} -f ex24_1.tmp
	-@${MPIEXEC} -n 3 ./ex24  | grep -v Process > ex24_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex24_1.out ex24_1.tmp) then true; \
	   else echo "Possible problem with ex24_1 3 processes, diffs above"; fi; ${RM} -f ex24_1.tmp
	-@${MPIEXEC} -n 4 ./ex24  | grep -v Process > ex24_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex24_1.out ex24_1.tmp) then true; \
	   else echo "Possible problem with ex24_1 4 processes, diffs above"; fi; ${RM} -f ex24_1.tmp
	-@${MPIEXEC} -n 5 ./ex24  | grep -v Process > ex24_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex24_1.out ex24_1.tmp) then true; \
	   else echo "Possible problem with ex24_1 5 processes, diffs above"; fi; ${RM} -f ex24_1.tmp
	-@${MPIEXEC} -n 6 ./ex24  | grep -v Process > ex24_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex24_1.out ex24_1.tmp) then true; \
	   else echo "Possible problem with ex24_1 6 processes, diffs above"; fi; ${RM} -f ex24_1.tmp


runex25:
	-@${MPIEXEC} -n 1 ./ex25  | grep -v Process > ex25_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex25_1.out ex25_1.tmp) then true; \
	   else echo "Possible problem with ex25_1 1 process, diffs above"; fi; ${RM} -f ex25_1.tmp
	-@${MPIEXEC} -n 2 ./ex25  | grep -v Process > ex25_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex25_1.out ex25_1.tmp) then true; \
	   else echo "Possible problem with ex25_1 2 processes, diffs above"; fi; ${RM} -f ex25_1.tmp
	-@${MPIEXEC} -n 3 ./ex25  | grep -v Process > ex25_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex25_1.out ex25_1.tmp) then true; \
	   else echo "Possible problem with ex25_1 3 processes, diffs above"; fi; ${RM} -f ex25_1.tmp
	-@${MPIEXEC} -n 4 ./ex25  | grep -v Process > ex25_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex25_1.out ex25_1.tmp) then true; \
	   else echo "Possible problem with ex25_1 4 processes, diffs above"; fi; ${RM} -f ex25_1.tmp
	-@${MPIEXEC} -n 5 ./ex25  | grep -v Process > ex25_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex25_1.out ex25_1.tmp) then true; \
	   else echo "Possible problem with ex25_1 5 processes, diffs above"; fi; ${RM} -f ex25_1.tmp
	-@${MPIEXEC} -n 6 ./ex25  | grep -v Process > ex25_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex25_1.out ex25_1.tmp) then true; \
	   else echo "Possible problem with ex25_1 6 processes, diffs above"; fi; ${RM} -f ex25_1.tmp

runex30_1:
	-@${MPIEXEC} -n 2 ./ex30 -bs 2 -block 0 -sliced_mat_type baij -alpha 10 -u0 0.1 > ex30_1.tmp; \
	   ${DIFF} output/ex30_1and2.out ex30_1.tmp || echo "Possible problem with ex30_1, diffs above"; \
	   ${RM} -f ex30_1.tmp
runex30_2:
	-@${MPIEXEC} -n 2 ./ex30 -bs 2 -block 1 -sliced_mat_type aij -alpha 10 -u0 0.1 > ex30_2.tmp; \
	   ${DIFF} output/ex30_1and2.out ex30_2.tmp || echo "Possible problem with ex30_2, diffs above"; \
	   ${RM} -f ex30_2.tmp
runex30_3:
	-@${MPIEXEC} -n 2 ./ex30 -bs 2 -block 0 -sliced_mat_type aij -alpha 10 -u0 0.1 > ex30_3.tmp; \
	   ${DIFF} output/ex30_3.out ex30_3.tmp || echo "Possible problem with ex30_3, diffs above"; \
	   ${RM} -f ex30_3.tmp
runex31:
	-@${MPIEXEC} -n 2 ./ex31 
runex32:
	-@${MPIEXEC} -n 2 ./ex32 > ex32_1.tmp; \
	  ${DIFF} output/ex32_1.out ex32_1.tmp || echo "Possible problem with ex32, diffs above"; \
	  ${RM} -f ex32_1.tmp

TESTEXAMPLES_C		  = ex1.PETSc runex1 ex1.rm ex4.PETSc runex4 ex4.rm ex24.PETSc runex24 ex24.rm ex25.PETSc \
                            runex25 ex25.rm ex31.PETSc runex31 ex31.rm ex32.PETSc runex32 ex32.rm
TESTEXAMPLES_C_X11	  = ex2.PETSc runex2 ex2.rm ex3.PETSc runex3 ex3.rm ex5.PETSc runex5 ex5.rm ex6.PETSc runex6 \
                            ex6.rm ex7.PETSc ex7.rm  ex20.PETSc runex20 runex20_2 ex20.rm ex14.PETSc runex14 ex14.rm \
                            ex13.PETSc runex13 ex13.rm ex23.PETSc runex23 ex23.rm
TESTEXAMPLES_FORTRAN	  = ex21f.PETSc ex21f.rm
TESTEXAMPLES_C_X11_MPIUNI = ex1.PETSc ex1.rm ex2.PETSc ex2.rm ex3.PETSc ex3.rm ex4.PETSc ex4.rm ex5.PETSc ex5.rm\
                            ex6.PETSc ex6.rm ex12.PETSc ex12.rm
TESTEXAMPLES_C_COMPLEX	  = 
TESTEXAMPLES_13		  = ex8.PETSc ex8.rm ex9.PETSc ex9.rm ex10.PETSc ex10.rm ex11.PETSc ex11.rm
TESTEXAMPLES_MATLAB	  = ex12.PETSc runex12 ex12.rm

include ${PETSC_DIR}/conf/test
