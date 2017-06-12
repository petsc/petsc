#requiresdefine   'PETSC_HAVE_ATTRIBUTEALIGNED'

CFLAGS	         =
FFLAGS	         =
CPPFLAGS         =
FPPFLAGS         =
EXAMPLESC        = ex1.c ex2.c ex1_nest.c
LOCDIR		 = src/ksp/ksp/examples/tutorials/network
MANSEC           = KSP

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules


ex1: ex1.o chkopts
	-${CLINKER} -o ex1 ex1.o ${PETSC_KSP_LIB}
	${RM} ex1.o

ex2: ex2.o chkopts
	-${CLINKER} -o ex2 ex2.o ${PETSC_KSP_LIB}
	${RM} ex2.o

ex1_nest: ex1_nest.o chkopts
	-${CLINKER} -o ex1_nest ex1_nest.o ${PETSC_KSP_LIB}
	${RM} ex1_nest.o
#---------------------------------------------------------------------------
runex1:
	-@${MPIEXEC} -n 1 ./ex1 -ksp_monitor_short > ex1_1.tmp 2>&1;   \
	   if (${DIFF} output/ex1_1.out ex1_1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex1, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex1_1.tmp
runex1_2:
	-@${MPIEXEC} -n 2 ./ex1 -petscpartitioner_type simple -ksp_converged_reason | ${SED} 's/iterations 2/iterations 3/g' > ex1_2.tmp 2>&1;   \
	   if (${DIFF} output/ex1_2.out ex1_2.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex1_2, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex1_2.tmp

runex2:
	-@${MPIEXEC} -n 1 ./ex2 -ksp_converged_reason > ex2.tmp 2>&1;   \
	   if (${DIFF} output/ex2.out ex2.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex2, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex2.tmp
runex2_2:
	-@${MPIEXEC} -n 2 ./ex2 -petscpartitioner_type simple -pc_type asm -sub_pc_type ilu -ksp_converged_reason > ex2_2.tmp 2>&1;   \
	   if (${DIFF} output/ex2_2.out ex2_2.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex2_2, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex2_2.tmp
runex2_3:
	-@${MPIEXEC} -n 4 ./ex2 -petscpartitioner_type simple -pc_type asm -sub_pc_type lu -sub_pc_factor_shift_type nonzero -ksp_converged_reason | ${SED} 's/iterations 36/iterations 35/g' > ex2.tmp 2>&1;   \
	   if (${DIFF} output/ex2_3.out ex2.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex2_3, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex2.tmp

runex2_graphindex:
	-@${MPIEXEC} -n 1 ./ex2 -n 20 -vertex_global_section_view -edge_global_section_view > ex2_graphindex.tmp 2>&1;   \
	   if (${DIFF} output/ex2_graphindex.out ex2_graphindex.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex2_graphindex, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex2_graphindex.tmp

runex2_graphindex_2:
	-@${MPIEXEC} -n 2 ./ex2 -petscpartitioner_type simple -n 20 -vertex_global_section_view -edge_global_section_view > ex2_graphindex_2.tmp 2>&1;   \
	   if (${DIFF} output/ex2_graphindex_2.out ex2_graphindex_2.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex2_graphindex_2, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex2_graphindex_2.tmp

runex1_nest:
	-@${MPIEXEC} -n 1 ./ex1_nest -ksp_converged_reason > ex1_nest.tmp 2>&1;   \
	   if (${DIFF} output/ex1_nest.out ex1_nest.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex1_nest, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex1_nest.tmp

runex1_nest_2:
	-@${MPIEXEC} -n 2 ./ex1_nest -petscpartitioner_type simple -ksp_converged_reason > ex1_nest_2.tmp 2>&1;   \
	   if (${DIFF} output/ex1_nest_2.out ex1_nest_2.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex1_nest_2, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex1_nest_2.tmp

TESTEXAMPLES_C_NOCOMPLEX_NOTSINGLE = ex1.PETSc runex1 runex1_2 ex1.rm \
                                     ex2.PETSc runex2 runex2_2 runex2_3 runex2_graphindex runex2_graphindex_2 ex2.rm \
                                     ex1_nest.PETSc runex1_nest runex1_nest_2 ex1_nest.rm

include ${PETSC_DIR}/lib/petsc/conf/test
