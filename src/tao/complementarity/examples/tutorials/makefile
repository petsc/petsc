ALL:

CFLAGS		 =
FFLAGS		 =
CPPFLAGS         =
FPPFLAGS         =
EXAMPLESC        = minsurf1.c blackscholes.c
EXAMPLESF        =
EXAMPLESCH       =
EXAMPLESFH       =
TUTORIALS        = minsurf1.c blackscholes.c
LOCDIR		 = src/tao/complementarity/examples/tutorials/
MANSEC		 =
DOCS		 =
DIRS		 =
CLEANFILES       = minsurf1 blackscholes

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

minsurf1: minsurf1.o chkopts
	-${CLINKER} -o minsurf1 minsurf1.o ${PETSC_TAO_LIB}
	${RM} minsurf1.o

blackscholes: blackscholes.o chkopts
	-${CLINKER} -o blackscholes blackscholes.o ${PETSC_TAO_LIB}
	${RM} blackscholes.o

runblackscholes:
	-@${MPIEXEC} -n 1 ./blackscholes -tao_monitor -tao_type ssils > blackscholes_1.tmp 2>&1; \
	${DIFF} output/blackscholes_1.out blackscholes_1.tmp || printf '${PWD}\nPossible problem with blackscholes_1 stdout, diffs above \n=========================================\n';\
	${RM} -f blackscholes_1.tmp

runblackscholes_2:
	-@${MPIEXEC} -n 1 ./blackscholes -tao_monitor -tao_type ssfls -tao_max_it 9 > blackscholes_2.tmp 2>&1; \
	${DIFF} output/blackscholes_2.out blackscholes_2.tmp || printf '${PWD}\nPossible problem with blackscholes_2 stdout, diffs above \n=========================================\n';\
	${RM} -f blackscholes_2.tmp

runblackscholes_3:
	-@${MPIEXEC} -n 1 ./blackscholes -tao_monitor -tao_type asils -tao_subset_type subvec > blackscholes_3.tmp 2>&1; \
	${DIFF} output/blackscholes_3.out blackscholes_3.tmp || printf '${PWD}\nPossible problem with blackscholes_3 stdout, diffs above \n=========================================\n';\
	${RM} -f blackscholes_3.tmp

runblackscholes_4:
	-@${MPIEXEC} -n 1 ./blackscholes -tao_monitor -tao_type asils -tao_subset_type mask > blackscholes_4.tmp 2>&1; \
	${DIFF} output/blackscholes_4.out blackscholes_4.tmp || printf '${PWD}\nPossible problem with blackscholes_4 stdout, diffs above \n=========================================\n';\
	${RM} -f blackscholes_4.tmp

runblackscholes_5:
	-@${MPIEXEC} -n 1 ./blackscholes -tao_monitor -tao_type asils -tao_subset_type matrixfree -pc_type jacobi -tao_max_it 5 > blackscholes_5.tmp 2>&1; \
	${DIFF} output/blackscholes_5.out blackscholes_5.tmp || printf '${PWD}\nPossible problem with blackscholes_5 stdout, diffs above \n=========================================\n';\
	${RM} -f blackscholes_5.tmp

runblackscholes_6:
	-@${MPIEXEC} -n 1 ./blackscholes -tao_monitor -tao_type asfls -tao_subset_type subvec -tao_max_it 9 > blackscholes_6.tmp 2>&1; \
	${DIFF} output/blackscholes_6.out blackscholes_6.tmp || printf '${PWD}\nPossible problem with blackscholes_6 stdout, diffs above \n=========================================\n';\
	${RM} -f blackscholes_6.tmp

runblackscholes_7:
	-@${MPIEXEC} -n 1 ./blackscholes -tao_monitor -tao_type asfls -tao_subset_type mask -tao_max_it 9 > blackscholes_7.tmp 2>&1; \
	${DIFF} output/blackscholes_7.out blackscholes_7.tmp || printf '${PWD}\nPossible problem with blackscholes_7 stdout, diffs above \n=========================================\n';\
	${RM} -f blackscholes_7.tmp

runminsurf1:
	-@${MPIEXEC} -n 1 ./minsurf1 -tao_monitor -tao_view  -tao_type ssils > minsurf1_1.tmp 2>&1; \
	${DIFF} output/minsurf1_1.out minsurf1_1.tmp || printf '${PWD}\nPossible problem with minsurf1_1 stdout, diffs above \n=========================================\n';\
	${RM} -f minsurf1_1.tmp

runminsurf1_2:
	-@${MPIEXEC} -n 1 ./minsurf1 -tao_monitor -tao_view -tao_type ssfls > minsurf1_2.tmp 2>&1; \
	${DIFF} output/minsurf1_2.out minsurf1_2.tmp || printf '${PWD}\nPossible problem with minsurf1_2 stdout, diffs above \n=========================================\n';\
	${RM} -f minsurf1_2.tmp

TESTEXAMPLES_C_NOTSINGLE = blackscholes.PETSc runblackscholes runblackscholes_2\
	                   runblackscholes_3 runblackscholes_4 printdot \
	                   runblackscholes_5 runblackscholes_6 \
	                   runblackscholes_7 blackscholes.rm \
                           minsurf1.PETSc runminsurf1 runminsurf1_2 minsurf1.rm
TESTEXAMPLES_C_X_MPIUNI  = blackscholes.PETSc runblackscholes runblackscholes_2\
	                   runblackscholes_3 runblackscholes_4 \
	                   runblackscholes_5 runblackscholes_6 \
	                   runblackscholes_7 blackscholes.rm \
                           minsurf1.PETSc runminsurf1 runminsurf1_2 minsurf1.rm


include ${PETSC_DIR}/lib/petsc/conf/test

