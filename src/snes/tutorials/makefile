-include ../../../petscdir.mk

EXAMPLESMATLAB   = ex5m.m ex29view.m
CLEANFILES       = ex5f90t

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex55: ex55k.o

#-------------------------------------------------------------------------

#  these tests are used by the makefile in PETSC_DIR for basic tests of the install and should not be removed
testex5f: ex5f.PETSc
	-@${MPIEXEC} -n 1 ${MPIEXEC_TAIL} ./ex5f -snes_rtol 1e-4 > ex5f_1.tmp 2>&1; \
        if (${DIFF} output/ex5f_1.testout ex5f_1.tmp > /dev/null 2>&1) then \
          echo "Fortran example src/snes/tutorials/ex5f run successfully with 1 MPI process"; \
        else \
          echo "Possible error running Fortran example src/snes/tutorials/ex5f with 1 MPI process"; \
          echo "See https://petsc.org/release/faq/";\
          cat ex5f_1.tmp; \
          touch ../../../check_error;\
        fi;  \
        ${RM} -f ex5f_1.tmp ;\
        ${MAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} ex5f.rm;
testex19: ex19.PETSc
	-@${MPIEXEC} -n 1 ${MPIEXEC_TAIL} ./ex19 -da_refine 3 -pc_type mg -ksp_type fgmres  > ex19_1.tmp 2>&1; \
        if (${DIFF} output/ex19_1.testout ex19_1.tmp > /dev/null 2>&1) then \
          echo "C/C++ example src/snes/tutorials/ex19 run successfully with 1 MPI process"; \
        else \
          echo "Possible error running C/C++ src/snes/tutorials/ex19 with 1 MPI process"; \
          echo "See https://petsc.org/release/faq/";\
          cat ex19_1.tmp;\
          touch ../../../check_error;\
        fi;\
        ${RM} -f ex19_1.tmp;
testex19_mpi:
	-@${MPIEXEC} -n 2 ${MPIEXEC_TAIL} ./ex19 -da_refine 3 -pc_type mg -ksp_type fgmres  > ex19_1.tmp 2>&1; \
        if (${DIFF} output/ex19_1.testout ex19_1.tmp > /dev/null 2>&1) then \
          echo "C/C++ example src/snes/tutorials/ex19 run successfully with 2 MPI processes"; \
        else \
          echo "Possible error running C/C++ src/snes/tutorials/ex19 with 2 MPI processes"; \
          echo "See https://petsc.org/release/faq/";\
          cat ex19_1.tmp; \
          touch ../../../check_error;\
        fi;\
        ${RM} -f ex19_1.tmp;
#use unpreconditioned norm because HYPRE device installations use different AMG parameters
runex19_hypre:
	-@${MPIEXEC} -n 2 ${MPIEXEC_TAIL} ./ex19 -da_refine 3 -snes_monitor_short -ksp_norm_type unpreconditioned -pc_type hypre > ex19_1.tmp 2>&1; \
          if (${DIFF} output/ex19_hypre.out ex19_1.tmp) then \
            echo "C/C++ example src/snes/tutorials/ex19 run successfully with HYPRE"; \
          else  \
            printf "${PWD}\nPossible problem with ex19 running with HYPRE, diffs above\n=========================================\n";\
            touch ../../../check_error;\
          fi; \
          ${RM} -f ex19_1.tmp
runex19_hypre_cuda:
	-@${MPIEXEC} -n 2 ${MPIEXEC_TAIL} ./ex19 -dm_vec_type cuda -dm_mat_type aijcusparse -da_refine 3 -snes_monitor_short -ksp_norm_type unpreconditioned -pc_type hypre > ex19_1.tmp 2>&1; \
	   if (${DIFF} output/ex19_hypre.out ex19_1.tmp) then \
           echo "C/C++ example src/snes/tutorials/ex19 run successfully with HYPRE/CUDA"; \
           else  \
             printf "${PWD}\nPossible problem with ex19 running with HYPRE/CUDA, diffs above\n=========================================\n"; \
             touch ../../../check_error;\
             fi; \
	   ${RM} -f ex19_1.tmp
runex19_hypre_hip:
	-@${MPIEXEC} -n 2 ${MPIEXEC_TAIL} ./ex19 -dm_vec_type hip -dm_mat_type aijhipsparse -da_refine 3 -snes_monitor_short -ksp_norm_type unpreconditioned -pc_type hypre > ex19_1.tmp 2>&1; \
	   if (${DIFF} output/ex19_hypre.out ex19_1.tmp) then \
           echo "C/C++ example src/snes/tutorials/ex19 run successfully with HYPRE/HIP"; \
           else \
             printf "${PWD}\nPossible problem with ex19 running with HYPRE/HIP, diffs above\n=========================================\n";\
             touch ../../../check_error;\
           fi; \
	   ${RM} -f ex19_1.tmp
runex19_cuda:
	-@${MPIEXEC} -n 1 ${MPIEXEC_TAIL} ./ex19 -snes_monitor -dm_mat_type seqaijcusparse -dm_vec_type seqcuda -pc_type gamg -ksp_monitor -mg_levels_ksp_max_it 1 > ex19_1.tmp 2>&1; \
	   if (${DIFF} output/ex19_cuda_1.out ex19_1.tmp) then \
           echo "C/C++ example src/snes/tutorials/ex19 run successfully with CUDA"; \
           else  \
             printf "${PWD}\nPossible problem with ex19 running with CUDA, diffs above\n=========================================\n"; \
             touch ../../../check_error;\
           fi; \
	   ${RM} -f ex19_1.tmp
runex19_hip:
	-@${MPIEXEC} -n 1 ${MPIEXEC_TAIL} ./ex19 -snes_monitor -dm_mat_type mpiaijhipsparse -dm_vec_type hip -pc_type gamg -ksp_monitor -mg_levels_ksp_max_it 1 > ex19_1.tmp 2>&1; \
	   if (${DIFF} output/ex19_hip_1.out ex19_1.tmp) then \
           echo "C/C++ example src/snes/tutorials/ex19 run successfully with HIP"; \
           else  \
             printf "${PWD}\nPossible problem with ex19 running with HIP, diffs above\n=========================================\n"; \
             touch ../../../check_error;\
           fi; \
	   ${RM} -f ex19_1.tmp
runex19_ml:
	-@${MPIEXEC} -n 2 ${MPIEXEC_TAIL} ./ex19 -da_refine 3 -snes_monitor_short -pc_type ml > ex19_1.tmp 2>&1; \
	   if (${DIFF} output/ex19_ml.out ex19_1.tmp) then  \
           echo "C/C++ example src/snes/tutorials/ex19 run successfully with ML"; \
           else \
             printf "${PWD}\nPossible problem with ex19 running with ML, diffs above\n=========================================\n"; \
             touch ../../../check_error;\
           fi; \
           ${RM} -f ex19_1.tmp
runex19_fieldsplit_mumps:
	-@${MPIEXEC} -n 2 ${MPIEXEC_TAIL} ./ex19 -pc_type fieldsplit -pc_fieldsplit_block_size 4 -pc_fieldsplit_type SCHUR -pc_fieldsplit_0_fields 0,1,2 -pc_fieldsplit_1_fields 3 -fieldsplit_0_pc_type lu -fieldsplit_1_pc_type lu -snes_monitor_short -ksp_monitor_short  -fieldsplit_0_pc_factor_mat_solver_type mumps -fieldsplit_1_pc_factor_mat_solver_type mumps > ex19_6.tmp 2>&1; \
	   if (${DIFF} output/ex19_fieldsplit_5.out ex19_6.tmp) then  \
           echo "C/C++ example src/snes/tutorials/ex19 run successfully with MUMPS"; \
           else  \
             printf "${PWD}\nPossible problem with ex19 running with MUMPS, diffs above\n=========================================\n";\
             touch ../../../check_error;\
           fi; \
           ${RM} -f ex19_6.tmp
runex19_superlu_dist:
	-@${MPIEXEC} -n 1 ${MPIEXEC_TAIL} ./ex19 -da_grid_x 20 -da_grid_y 20 -pc_type lu -pc_factor_mat_solver_type superlu_dist > ex19.tmp 2>&1; \
	   if (${DIFF} output/ex19_superlu.out ex19.tmp) then \
           echo "C/C++ example src/snes/tutorials/ex19 run successfully with SuperLU_DIST"; \
           else  \
             printf "${PWD}\nPossible problem with ex19 running with SuperLU_DIST, diffs above\n=========================================\n"; \
             touch ../../../check_error;\
           fi; \
	   ${RM} -f ex19.tmp
runex19_suitesparse:
	-@${MPIEXEC} -n 1 ${MPIEXEC_TAIL} ./ex19 -da_refine 3 -snes_monitor_short -pc_type lu -pc_factor_mat_solver_type umfpack > ex19_1.tmp 2>&1; \
	   if (${DIFF} output/ex19_suitesparse.out ex19_1.tmp) then \
           echo "C/C++ example src/snes/tutorials/ex19 run successfully with SuiteSparse"; \
           else \
             printf "${PWD}\nPossible problem with ex19 running with SuiteSparse, diffs above\n=========================================\n"; \
             touch ../../../check_error;\
           fi; \
	   ${RM} -f ex19_1.tmp
runex3k_kokkos: ex3k.PETSc
	-@KOKKOS_DISABLE_WARNINGS=1 ${MPIEXEC} -n 2 ${MPIEXEC_TAIL} ./ex3k -view_initial -dm_vec_type kokkos -dm_mat_type aijkokkos -use_gpu_aware_mpi 0 -snes_monitor > ex3k_1.tmp 2>&1 ;\
	if (${DIFF} output/ex3k_1.out ex3k_1.tmp) then \
          echo "C/C++ example src/snes/tutorials/ex3k run successfully with Kokkos Kernels"; \
        else \
          printf "${PWD}\nPossible problem with ex3k running with Kokkos Kernels, diffs above\n=========================================\n"; \
          touch ../../../check_error;\
        fi; \
	${RM} -f ex3k_1.tmp

