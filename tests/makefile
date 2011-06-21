LOCDIR=tests/
EXAMPLESC = test_link1.c linesearch1.c
TESTEXAMPLES_C = test_link1.tao runtest_link1 test_link1.rm clean \
         test_linesearch1.tao runtest_linesearch1 test_linesearch1.rm \
         test_lmvmmat1.tao runtest_lmvmmat1 test_lmvmmat1.rm \
         test_approxmat1.tao runtest_approxmat1 test_approxmat1.rm \
         test_rosenbrock1.tao runtest_rosenbrock1 test_rosenbrock1.rm \
         test_jbearing2.tao runtest_jbearing2 test_jbearing2.rm \
	 test_eptorsion1.tao runtest_eptorsion1 test_eptorsion2.rm \
	 test_eptorsion2.tao runtest_eptorsion2 test_eptorsion2.rm \
	 test_minsurf1.tao runtest_minsurf1 test_minsurf1.rm \
	 test_minsurf2.tao runtest_minsurf2 test_minsurf2.rm \
	 test_plate2.tao runtest_plate2 test_plate2.rm \
         test_jbearingdm.tao runtest_jbearingdm test_jbearingdm.rm \
         test_jbearingdmf.tao runtest_jbearingdmf test_jbearingdmf.rm \
	 clean 
TESTEXAMPLES_F=	 test_eptorsion2f.tao runtest_eptorsion2f test_eptorsion2f.rm \
	 test_plate2f.tao runtest_plate2f test_plate2f.rm \
         test_rosenbrock1f.tao runtest_rosenbrock1f test_rosenbrock1f.rm  \
	 clean 

include ${TAO_DIR}/conf/tao_base

test_link1: test_link1.o tao_chkopts

test_link1f: test_link1f.o tao_chkopts
	-${FLINKER} -o test_link1f test_link1f.o ${TAO_LIB_FORTRAN} ${TAO_LIB} ${PETSC_LIB_FORTRAN} ${PETSC_LIB}
	${RM} test_link1f.o


test_subvec1: test_subvec1.o tao_chkopts
	-${CLINKER} -o test_subvec1 test_subvec1.o ${TAO_LIB} ${PETSC_LIB}
	${RM} test_subvec1.o

test_submat1: test_submat1.o tao_chkopts
	-${CLINKER} -o test_submat1 test_submat1.o ${TAO_LIB} ${PETSC_LIB}
	${RM} test_submat1.o


test_linesearch1: test_linesearch1.o tao_chkopts
	-${CLINKER} -o test_linesearch1 test_linesearch1.o ${TAO_LIB} ${PETSC_LIB}
	${RM} test_linesearch1.o

test_lmvmmat1: test_lmvmmat1.o tao_chkopts
	-${CLINKER} -o test_lmvmmat1 test_lmvmmat1.o ${TAO_LIB} ${PETSC_LIB}
	${RM} test_lmvmmat1.o

test_approxmat1: test_approxmat1.o tao_chkopts
	-${CLINKER} -o test_approxmat1 test_approxmat1.o ${TAO_LIB} ${PETSC_LIB}
	${RM} test_approxmat1.o

test_rosenbrock1: test_rosenbrock1.o tao_chkopts
	-${CLINKER} -o test_rosenbrock1 test_rosenbrock1.o ${TAO_LIB} ${PETSC_LIB}
	${RM} test_rosenbrock1.o

test_rosenbrock1f: test_rosenbrock1f.o tao_chkopts
	-${FLINKER} -o  test_rosenbrock1f test_rosenbrock1f.o ${TAO_LIB} ${PETSC_LIB}
	${RM} test_rosenbrock1f.o

test_jbearing2: test_jbearing2.o tao_chkopts
	-${CLINKER} -o test_jbearing2 test_jbearing2.o ${TAO_LIB} ${PETSC_LIB}
	${RM} test_jbearing2.o


test_jbearingdm: test_jbearingdm.o tao_chkopts
	-${CLINKER} -o test_jbearingdm test_jbearingdm.o ${TAO_LIB} ${PETSC_LIB}
	${RM} test_jbearingdm.o


test_jbearingdmf: test_jbearingdmf.o tao_chkopts
	-${FLINKER} -o test_jbearingdmf test_jbearingdm.o ${TAO_LIB} ${PETSC_LIB}
	${RM} test_jbearingdmf.o

test_plate2: test_plate2.o tao_chkopts
	-${CLINKER} -o test_plate2 test_plate2.o ${TAO_LIB} ${PETSC_LIB}
	${RM} test_plate2.o

test_plate2f: test_plate2f.o tao_chkopts
	-${FLINKER} -o test_plate2f test_plate2f.o ${TAO_LIB} ${PETSC_LIB}
	${RM} test_plate2f.o

test_eptorsion1: test_eptorsion1.o tao_chkopts
	-${CLINKER} -o test_eptorsion1 test_eptorsion1.o ${TAO_LIB} ${PETSC_LIB}
	${RM} test_eptorsion1.o

test_eptorsion2: test_eptorsion2.o tao_chkopts
	-${CLINKER} -o test_eptorsion2 test_eptorsion2.o ${TAO_LIB} ${PETSC_LIB}
	${RM} test_eptorsion2.o

test_eptorsion2f: test_eptorsion2f.o tao_chkopts
	-${FLINKER} -o test_eptorsion2f test_eptorsion2f.o ${TAO_LIB} ${PETSC_LIB}
	${RM} test_eptorsion2f.o

test_minsurf1: test_minsurf1.o tao_chkopts
	-${CLINKER} -o test_minsurf1 test_minsurf1.o ${TAO_LIB} ${PETSC_LIB}
	${RM} test_minsurf1.o

test_minsurf3: test_minsurf3.o tao_chkopts
	-${CLINKER} -o test_minsurf3 test_minsurf3.o ${TAO_LIB} ${PETSC_LIB}
	${RM} test_minsurf3.o

test_dgesv: test_dgesv.o tao_chkopts
	-${CLINKER} -o test_dgesv test_dgesv.o  -llapack -lblas

test_minsurf2: test_minsurf2.o tao_chkopts
	-${CLINKER} -o test_minsurf2 test_minsurf2.o ${TAO_LIB} ${PETSC_LIB}
	${RM} test_minsurf2.o

test_bratu2: test_bratu2.o tao_chkopts
	-${CLINKER} -o test_bratu2 test_bratu2.o ${TAO_LIB} ${PETSC_LIB}
	${RM} test_bratu2.o

test_chwirut1: test_chwirut1.o tao_chkopts
	-${CLINKER} -o test_chwirut1 test_chwirut1.o ${TAO_LIB} ${PETSC_LIB}
	${RM} test_chwirut1.o

test_copper1: test_copper1.o tao_chkopts
	-${CLINKER} -o test_copper1 test_copper1.o ${TAO_LIB} ${PETSC_LIB}
	${RM} test_copper1.o

test_chwirut1f: test_chwirut1f.o tao_chkopts
	-${FLINKER} -o  test_chwirut1f test_chwirut1f.o ${TAO_LIB} ${PETSC_LIB}
	${RM} test_chwirut1f.o

test_toy: test_toy.o tao_chkopts
	-${CLINKER} -o test_toy test_toy.o ${TAO_LIB} ${PETSC_LIB}
	${RM} test_toy.o

runtest_link1: 
	-@${MPIEXEC} -np 1 ./test_link1 > test_link1_1.tmp 2>&1; \
	if (${DIFF} output/test_link1_1.out test_link1_1.tmp) then true;\
        else echo "Possible problem with test_link1_1, diffs above"; fi; \
	${RM} -f test_link1_1.tmp

runtest_linesearch1: 
	-@${MPIEXEC} -np 1 ./test_linesearch1 > test_linesearch1_1.tmp 2>&1; \
	if (${DIFF} output/test_linesearch1_1.out test_linesearch1_1.tmp) then true;\
        else echo "Possible problem with test_linesearch1_1, diffs above"; fi; \
	${RM} -f test_linesearch1_1.tmp

runtest_lmvmmat1: 
	-@${MPIEXEC} -np 1 ./test_lmvmmat1 > test_lmvmmat1_1.tmp 2>&1; \
	if (${DIFF} output/test_lmvmmat1_1.out test_lmvmmat1_1.tmp) then true;\
        else echo "Possible problem with test_lmvmmat1_1, diffs above"; fi; \
	${RM} -f test_lmvmmat1_1.tmp


runtest_rosenbrock1: 
	-@${MPIEXEC} -np 1 ./test_rosenbrock1 > test_rosenbrock1_1.tmp 2>&1; \
	if (${DIFF} output/test_rosenbrock1_1.out test_rosenbrock1_1.tmp) then true;\
        else echo "Possible problem with test_rosenbrock1_1, diffs above"; fi; \
	${RM} -f test_rosenbrock1_1.tmp


runtest_jbearing2: 
	-@${MPIEXEC} -np 2 ./test_jbearing2 > test_jbearing2_1.tmp 2>&1; \
	if (${DIFF} output/test_jbearing2_1.out test_jbearing2_1.tmp) then true;\
        else echo "Possible problem with test_jbearing2_1, diffs above"; fi; \
	${RM} -f test_jbearing2_1.tmp

runtest_jbearingdm: 
	-@${MPIEXEC} -np dm ./test_jbearingdm > test_jbearingdm_1.tmp dm>&1; \
	if (${DIFF} output/test_jbearingdm_1.out test_jbearingdm_1.tmp) then true;\
        else echo "Possible problem with test_jbearingdm_1, diffs above"; fi; \
	${RM} -f test_jbearingdm_1.tmp

runtest_plate2: 
	-@${MPIEXEC} -np 2 ./test_plate2 > test_plate2_1.tmp 2>&1; \
	if (${DIFF} output/test_plate2_1.out test_plate2_1.tmp) then true;\
        else echo "Possible problem with test_plate2_1, diffs above"; fi; \
	${RM} -f test_plate2_1.tmp


runtest_plate2f: 
	-@${MPIEXEC} -np 2 ./test_plate2f > test_plate2f_1.tmp 2>&1; \
	if (${DIFF} output/test_plate2f_1.out test_plate2f_1.tmp) then true;\
        else echo "Possible problem with test_plate2f_1, diffs above"; fi; \
	${RM} -f test_plate2f_1.tmp

runtest_eptorsion1: 
	-@${MPIEXEC} -np 1 ./test_eptorsion1 -tao_smonitor -tao_method tao_lmvm > test_eptorsion1_1.tmp 2>&1; \
	if (${DIFF} output/test_eptorsion1_1.out test_eptorsion1_1.tmp) then true;\
        else echo "Possible problem with test_eptorsion1_1, diffs above"; fi; \
	${RM} -f test_eptorsion1_1.tmp

runtest_eptorsion1_2:
	-@${MPIEXEC} -np 1 ./test_eptorsion1 -tao_smonitor -tao_method tao_nls \
	 > test_eptorsion1_2.tmp 2>&1; \
	if (${DIFF} output/test_eptorsion1_2.out test_eptorsion1_2.tmp) then true;\
	else echo "Possible problem with test_eptorsion1_2, diffs above"; fi;\
	${RM} -f test_eptorsion1_2.tmp

runtest_eptorsion1_3:
	-@${MPIEXEC} -np 1 ./test_eptorsion1 -tao_smonitor -tao_method tao_cg \
        -tao_cg_type prp > test_eptorsion1_3.tmp 2>&1; \
	if (${DIFF} output/test_eptorsion1_3.out test_eptorsion1_3.tmp) then true;\
	else echo "Possible problem with test_eptorsion1_3, diffs above"; fi;\
	${RM} -f test_eptorsion1_3.tmp

runtest_eptorsion1_4:
	-@${MPIEXEC} -np 1 ./test_eptorsion1 -tao_smonitor -tao_method tao_ntr \
         > test_eptorsion1_4.tmp 2>&1; \
	if (${DIFF} output/test_eptorsion1_4.out test_eptorsion1_4.tmp) then true;\
	else echo "Possible problem with test_eptorsion1_4, diffs above"; fi;\
	${RM} -f test_eptorsion1_4.tmp



runtest_eptorsion2: 
	-@${MPIEXEC} -np 1 ./test_eptorsion2 -tao_smonitor -tao_method tao_nls \
	> test_eptorsion2_1.tmp 2>&1; \
	if (${DIFF} output/test_eptorsion2_1.out test_eptorsion2_1.tmp) then true;\
        else echo "Possible problem with test_eptorsion2_1, diffs above"; fi; \
	${RM} -f test_eptorsion2_1.tmp

runtest_eptorsion2_2: 
	-@${MPIEXEC} -np 2 ./test_eptorsion2 -tao_smonitor -tao_method tao_nls \
	 > test_eptorsion2_2.tmp 2>&1; \
	if (${DIFF} output/test_eptorsion2_2.out test_eptorsion2_2.tmp) then true;\
        else echo "Possible problem with test_eptorsion2_2, diffs above"; fi; \
	${RM} -f test_eptorsion2_2.tmp

runtest_eptorsion2_3: 
	-@${MPIEXEC} -np 1 ./test_eptorsion2 -tao_smonitor -tao_method tao_ntr \
	> test_eptorsion2_3.tmp 2>&1; \
	if (${DIFF} output/test_eptorsion2_3.out test_eptorsion2_3.tmp) then true;\
        else echo "Possible problem with test_eptorsion2_3, diffs above"; fi; \
	${RM} -f test_eptorsion2_3.tmp

runtest_eptorsion2_4: 
	-@${MPIEXEC} -np 2 ./test_eptorsion2 -mx 16 -my 16 -tao_smonitor \
	-tao_method tao_ntr > test_eptorsion2_4.tmp 2>&1; \
	if (${DIFF} output/test_eptorsion2_4.out test_eptorsion2_4.tmp) then true;\
        else echo "Possible problem with test_eptorsion2_4, diffs above"; fi; \
	${RM} -f test_eptorsion2_4.tmp

runtest_eptorsion2_5: 
	-@${MPIEXEC} -np 1 ./test_eptorsion2 -tao_method tao_lmvm -tao_smonitor \
	> test_eptorsion2_5.tmp 2>&1; \
	if (${DIFF} output/test_eptorsion2_5.out test_eptorsion2_5.tmp) then true;\
        else echo "Possible problem with test_eptorsion2_5, diffs above"; fi; \
	${RM} -f test_eptorsion2_5.tmp

runtest_eptorsion2_6: 
	-@${MPIEXEC} -np 3 ./test_eptorsion2 -tao_method tao_lmvm -mx 16 -my 16 \
	 -tao_smonitor  test_eptorsion2_6.tmp 2>&1; \
	if (${DIFF} output/test_eptorsion2_6.out test_eptorsion2_6.tmp) then true;\
        else echo "Possible problem with test_eptorsion2_6, diffs above"; fi; \
	${RM} -f test_eptorsion2_6.tmp


runtest_eptorsion2f: 
	-@${MPIEXEC} -np 1 ./test_eptorsion2f -tao_smonitor -tao_method tao_nls \
	 > test_eptorsion2f_1.tmp 2>&1; \
	if (${DIFF} output/test_eptorsion2f_1.out test_eptorsion2f_1.tmp) then true;\
        else echo "Possible problem with test_eptorsion2f_1, diffs above"; fi; \
	${RM} -f test_eptorsion2f_1.tmp

runtest_eptorsion2f_1: 
	-@${MPIEXEC} -np 2 ./test_eptorsion2f -tao_smonitor -tao_method tao_nls \
	 > test_eptorsion2f_1.tmp 2>&1; \
	if (${DIFF} output/test_eptorsion2f_1.out test_eptorsion2f_1.tmp) then true;\
        else echo "Possible problem with test_eptorsion2f_1, diffs above"; fi; \
	${RM} -f test_eptorsion2f_1.tmp

runtest_eptorsion2f_2: 
	-@${MPIEXEC} -np 3 ./test_eptorsion2f -tao_smonitor -tao_method tao_lmvm \
	-mx 16 -my 16  > test_eptorsion2f_1.tmp 2>&1; \
	if (${DIFF} output/test_eptorsion2f_1.out test_eptorsion2f_1.tmp) then true;\
        else echo "Possible problem with test_eptorsion2f_1, diffs above"; fi; \
	${RM} -f test_eptorsion2f_1.tmp


runtest_minsurf1: 
	-@${MPIEXEC} -np 1 ./test_minsurf1 > test_minsurf1_1.tmp 2>&1; \
	if (${DIFF} output/test_minsurf1_1.out test_minsurf1_1.tmp) then true;\
        else echo "Possible problem with test_minsurf1_1, diffs above"; fi; \
	${RM} -f test_minsurf1_1.tmp


runtest_minsurf2: 
	-@${MPIEXEC} -np 2 ./test_minsurf2 > test_minsurf2_1.tmp 2>&1; \
	if (${DIFF} output/test_minsurf2_1.out test_minsurf2_1.tmp) then true;\
        else echo "Possible problem with test_minsurf2_1, diffs above"; fi; \
	${RM} -f test_minsurf2_1.tmp


runtest_bratu2: 
	-@${MPIEXEC} -np 2 ./test_bratu2 > test_bratu2_1.tmp 2>&1; \
	if (${DIFF} output/test_bratu2_1.out test_bratu2_1.tmp) then true;\
        else echo "Possible problem with test_bratu2_1, diffs above"; fi; \
	${RM} -f test_bratu2_1.tmp

runtest_chwirut1: 
	-@${MPIEXEC} -np 2 ./test_chwirut1 > test_chwirut1_1.tmp 2>&1; \
	if (${DIFF} output/test_chwirut1_1.out test_chwirut1_1.tmp) then true;\
        else echo "Possible problem with test_chwirut1_1, diffs above"; fi; \
	${RM} -f test_chwirut1_1.tmp


