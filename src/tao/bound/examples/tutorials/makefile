ALL:

CFLAGS		 =
FFLAGS		 =
CPPFLAGS         =
FPPFLAGS         =
TUTORIALS        = plate2.c jbearing2.c plate2f.F90
EXAMPLESC        = plate2.c jbearing2.c
EXAMPLECH        =
EXAMPLESF        = plate2f.F90
EXAMPLESFH       = plate2f.h
LOCDIR		 = src/tao/bound/examples/tutorials/
MANSEC		 =
DOCS		 =
DIRS		 =
CLEANFILES       = plate2 jbearing2 plate2f jbearing2f

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

plate2: plate2.o chkopts
	-${CLINKER} -o plate2 plate2.o ${PETSC_TAO_LIB}
	${RM} plate2.o

plate2f: plate2f.o chkopts
	-${FLINKER} -o plate2f plate2f.o ${PETSC_TAO_LIB}
	${RM} plate2f.o

jbearing2: jbearing2.o chkopts
	-${CLINKER} -o jbearing2 jbearing2.o ${PETSC_TAO_LIB}
	${RM} jbearing2.o


runplate2:
	-@${MPIEXEC} -n 1 ./plate2 -tao_smonitor -mx 8 -my 6 -bmx 3 -bmy 3 -bheight 0.2 -tao_type tron > plate2_1.tmp 2>&1;\
        ${DIFF} output/plate2_1.out plate2_1.tmp || printf '${PWD}\nPossible problem with plate2_1 stdout, diffs above \n=========================================\n';\
        ${RM} -f plate2_1.tmp

runplate2_2:
	-@${MPIEXEC} -n 2 ./plate2 -tao_smonitor -mx 8 -my 8 -bmx 2 -bmy 5 -bheight 0.3 -tao_type blmvm -tao_gttol 1.e-5  > plate2_2.tmp 2>&1;\
        ${DIFF} output/plate2_2.out plate2_2.tmp || printf '${PWD}\nPossible problem with plate2_2 stdout, diffs above \n=========================================\n';\
        ${RM} -f plate2_2.tmp

runplate2_3:
	-@${MPIEXEC} -n 3 ./plate2 -tao_smonitor -mx 8 -my 12 -bmx 4 -bmy 10 -bheight 0.1 -tao_type tron -tao_gttol 1.e-5 > plate2_3.tmp 2>&1;\
        ${DIFF} output/plate2_3.out plate2_3.tmp || printf '${PWD}\nPossible problem with plate2_3 stdout, diffs above \n=========================================\n';\
        ${RM} -f plate2_3.tmp

runplate2_4:
	-@${MPIEXEC} -n 3 ./plate2 -tao_smonitor -mx 8 -my 12 -bmx 4 -bmy 10 -bheight 0.1 -tao_subset_type mask -tao_type tron -tao_gttol 1.e-5 > plate2_4.tmp 2>&1;\
        ${DIFF} output/plate2_4.out plate2_4.tmp || printf '${PWD}\nPossible problem with plate2_4 stdout, diffs above \n=========================================\n';\
        ${RM} -f plate2_4.tmp

runplate2_5:
	-@${MPIEXEC} -n 3 ./plate2 -tao_smonitor -mx 8 -my 12 -bmx 4 -bmy 10 -bheight 0.1 -tao_subset_type matrixfree -matrixfree -pc_type none -tao_type tron  -tao_gttol 1.e-5 > plate2_5.tmp 2>&1;\
        ${DIFF} output/plate2_5.out plate2_5.tmp || printf '${PWD}\nPossible problem with plate2_5 stdout, diffs above \n=========================================\n';\
        ${RM} -f plate2_5.tmp

runplate2_6:
	-@${MPIEXEC} -n 3 ./plate2 -tao_smonitor -mx 8 -my 12 -bmx 4 -bmy 10 -bheight 0.1 -tao_subset_type matrixfree -matrixfree  -tao_type blmvm -tao_gttol 1.e-5 > plate2_6.tmp 2>&1;\
        ${DIFF} output/plate2_6.out plate2_6.tmp || printf '${PWD}\nPossible problem with plate2_6 stdout, diffs above \n=========================================\n';\
        ${RM} -f plate2_6.tmp

runplate2_7:
	-@${MPIEXEC} -n 3 ./plate2 -tao_smonitor -mx 8 -my 12 -bmx 4 -bmy 10 -bheight 0.1 -tao_subset_type matrixfree  -pc_type none -tao_type gpcg -tao_gttol 1.e-5 > plate2_7.tmp 2>&1;\
        ${DIFF} output/plate2_7.out plate2_7.tmp || printf '${PWD}\nPossible problem with plate2_7 stdout, diffs above \n=========================================\n';\
        ${RM} -f plate2_7.tmp

runplate2f:
	-@${MPIEXEC} -n 1 ./plate2f -tao_smonitor -mx 8 -my 6 -bmx 3 -bmy 3 -bheight 0.2 -tao_type blmvm -tao_gttol 1.e-4 | sort -b  > plate2f_1.tmp 2>&1;\
	sort -b output/plate2f_1.out | ${DIFF} - plate2f_1.tmp || printf '${PWD}\nPossible problem with plate2f_1 stdout, diffs above \n=========================================\n';\
	${RM} -f plate2f_1.tmp

runplate2f_2:
	-@${MPIEXEC} -n 2 ./plate2f -tao_smonitor -mx 8 -my 6 -bmx 3 -bmy 3 -bheight 0.2 -tao_type blmvm -tao_gttol 1.e-4 | sort -b  > plate2f_2.tmp 2>&1;\
	sort -b  output/plate2f_2.out | ${DIFF} - plate2f_2.tmp || printf '${PWD}\nPossible problem with plate2f_2 stdout, diffs above \n=========================================\n';\
	${RM} -f plate2f_2.tmp

runjbearing2:
	-@${MPIEXEC} -n 1 ./jbearing2 -tao_smonitor -mx 8 -my 12 -tao_type tron -tao_gttol 1.e-5  > jbearing2_1.tmp 2>&1;\
	${DIFF} output/jbearing2_1.out jbearing2_1.tmp || printf '${PWD}\nPossible problem with jbearing2_1 stdout, diffs above \n=========================================\n';\
	${RM} -f jbearing2_1.tmp

runjbearing2_2:
	-@${MPIEXEC} -n 2 ./jbearing2 -tao_smonitor -mx 50 -my 50 -ecc 0.99 -tao_type gpcg -tao_gttol 1.e-5 > jbearing2_2.tmp 2>&1;\
	${DIFF} output/jbearing2_2.out jbearing2_2.tmp || printf '${PWD}\nPossible problem with jbearing2_2 stdout, diffs above \n=========================================\n';\
	${RM} -f jbearing2_2.tmp

runjbearing2_3:
	-@${MPIEXEC} -n 2 ./jbearing2 -tao_smonitor -mx 10 -my 16 -ecc 0.9 -tao_type bqpip -tao_gatol 1.e-4 > jbearing2_3.tmp 2>&1;\
	${DIFF} output/jbearing2_3.out jbearing2_3.tmp || printf '${PWD}\nPossible problem with jbearing2_3 stdout, diffs above \n=========================================\n';\
	${RM} -f jbearing2_3.tmp






TESTEXAMPLES_C_NOTSINGLE        = plate2.PETSc runplate2 runplate2_2 runplate2_3 runplate2_4 runplate2_5 runplate2_6 runplate2_7 plate2.rm printdot \
	                          jbearing2.PETSc runjbearing2 runjbearing2_2 runjbearing2_3 jbearing2.rm 
TESTEXAMPLES_C_X_MPIUNI         = plate2.PETSc runplate2 plate2.rm jbearing2.PETSc runjbearing2 jbearing2.rm
TESTEXAMPLES_FORTRAN_NOTSINGLE  = plate2f.PETSc runplate2f runplate2f_2 plate2f.rm
TESTEXAMPLES_FORTRAN_MPIUNI     = plate2f.PETSc runplate2f plate2f.rm

include ${PETSC_DIR}/lib/petsc/conf/test
