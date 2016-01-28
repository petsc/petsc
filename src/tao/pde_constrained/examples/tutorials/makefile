CFLAGS           =
FFLAGS		 =
CPPFLAGS         =
FPPFLAGS         =
LOCDIR		 = src/tao/pde_constrained/examples/tutorials/
MANSEC           =
EXAMPLESC	 = elliptic.c parabolic.c hyperbolic.c
TUTORIALS        = elliptic.c parabolic.c hyperbolic.c
EXAMPLESF	 =
EXAMPLESCH	 =
EXAMPLESFH       =
EXAMPLESMATLAB   =
CLEANFILES       = elliptic parabolic hyperbolic

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

parabolic: parabolic.o chkopts
	-${CLINKER} -o parabolic parabolic.o ${PETSC_TAO_LIB}
	${RM} parabolic.o

hyperbolic: hyperbolic.o chkopts
	-${CLINKER} -o hyperbolic hyperbolic.o ${PETSC_TAO_LIB}
	${RM} hyperbolic.o

elliptic: elliptic.o chkopts
	-${CLINKER} -o elliptic elliptic.o ${PETSC_TAO_LIB}
	${RM} elliptic.o

runelliptic:
	-@${MPIEXEC} -n 1 ./elliptic -tao_cmonitor -ns 1 -tao_type lcl -tao_gatol 1.e-3 > elliptic_1.tmp 2>&1;\
        ${DIFF} output/elliptic_1.out elliptic_1.tmp || printf '${PWD}\nPossible problem with elliptic_1 stdout, diffs above \n=========================================\n';\
        ${RM} -f elliptic_1.tmp

runelliptic_2:
	-@${MPIEXEC} -n 1 ./elliptic -tao_cmonitor -tao_type lcl -tao_max_it 10 -use_ptap -use_lrc -ns 1  -tao_gttol 1.e-5 > elliptic_2.tmp 2>&1;\
        ${DIFF} output/elliptic_2.out elliptic_2.tmp || printf '${PWD}\nPossible problem with elliptic_2 stdout, diffs above \n=========================================\n';\
        ${RM} -f elliptic_2.tmp

runparabolic:
	-@${MPIEXEC} -n 1 ./parabolic -tao_cmonitor -tao_type lcl -ns 1  -tao_gatol 1.e-4 -ksp_max_it 30 > parabolic_1.tmp 2>&1;\
        ${DIFF} output/parabolic_1.out parabolic_1.tmp || printf '${PWD}\nPossible problem with parabolic_1 stdout, diffs above \n=========================================\n';\
        ${RM} -f parabolic_1.tmp

runhyperbolic:
	-@${MPIEXEC} -n 1 ./hyperbolic -tao_cmonitor -tao_max_funcs 10 -tao_type lcl > hyperbolic_1.tmp 2>&1;\
        ${DIFF} output/hyperbolic_1.out hyperbolic_1.tmp || printf '${PWD}\nPossible problem with hyperbolic_1 stdout, diffs above \n=========================================\n';\
        ${RM} -f hyperbolic_1.tmp

TESTEXAMPLES_C            = 
TESTEXAMPLES_C_NOTSINGLE  = hyperbolic.PETSc runhyperbolic hyperbolic.rm elliptic.PETSc runelliptic runelliptic_2 runelliptic elliptic.rm parabolic.PETSc runparabolic parabolic.rm

include ${PETSC_DIR}/lib/petsc/conf/test
