
CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR   = src/sys/classes/draw/examples/tests/
EXAMPLESC       = ex1.c ex2.c ex3.c ex4.c ex6.c ex7.c ex9.c ex10.c ex11.c ex12.c
EXAMPLESF       = ex4f.F ex5f.F
MANSEC          = Sys
SUBMANSEC       = Draw

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex1: ex1.o  chkopts
	-${CLINKER} -o ex1 ex1.o  ${PETSC_SYS_LIB}
	${RM} -f ex1.o

ex2: ex2.o  chkopts
	-${CLINKER} -o ex2 ex2.o  ${PETSC_SYS_LIB}
	${RM} -f ex2.o

ex3: ex3.o  chkopts
	-${CLINKER} -o ex3 ex3.o  ${PETSC_SYS_LIB}
	${RM} -f ex3.o

ex4: ex4.o  chkopts
	-${CLINKER} -o ex4 ex4.o  ${PETSC_SYS_LIB}
	${RM} -f ex4.o

ex4f: ex4f.o  chkopts
	-${FLINKER} -o ex4f ex4f.o  ${PETSC_SYS_LIB}
	${RM} -f ex4f.o

ex5f: ex5f.o  chkopts
	-${FLINKER} -o ex5f ex5f.o ${PETSC_SYS_LIB}
	${RM} -f ex5f.o

ex6: ex6.o  chkopts
	-${CLINKER} -o ex6 ex6.o  ${PETSC_SYS_LIB}
	${RM} -f ex6.o

ex7: ex7.o  chkopts
	-${CLINKER} -o ex7 ex7.o  ${PETSC_SYS_LIB}
	${RM} -f ex7.o

ex8: ex8.o  chkopts
	-${CLINKER} -o ex8 ex8.o  ${PETSC_SYS_LIB}
	${RM} -f ex8.o

ex9: ex9.o  chkopts
	-${CLINKER} -o ex9 ex9.o  ${PETSC_SYS_LIB}
	${RM} -f ex9.o

ex10: ex10.o  chkopts
	-${CLINKER} -o ex10 ex10.o  ${PETSC_SYS_LIB}
	${RM} -f ex10.o

ex11: ex11.o  chkopts
	-${CLINKER} -o ex11 ex11.o  ${PETSC_SYS_LIB}
	${RM} -f ex11.o

ex12: ex12.o  chkopts
	-${CLINKER} -o ex12 ex12.o  ${PETSC_SYS_LIB}
	${RM} -f ex12.o
#------------------------------------------------------------------------
runex1:
	-@${MPIEXEC} -n 1 ./ex1
runex2:
	-@${MPIEXEC} -n 1 ./ex2
runex3:
	-@${MPIEXEC} -n 1 ./ex3
runex4:
	-@${MPIEXEC} -n 1 ./ex4
runex4_db:
	-@${MPIEXEC} -n 1 ./ex4 -draw_double_buffer 0
runex4_dv:
	-@${MPIEXEC} -n 1 ./ex4 -draw_virtual
runex4_df:
	-@${MPIEXEC} -n 1 ./ex4 -draw_fast
runex6:
	-@${MPIEXEC} -n 1 ./ex6 -draw_cmap hue
	-@${MPIEXEC} -n 1 ./ex6 -draw_cmap gray
	-@${MPIEXEC} -n 1 ./ex6 -draw_cmap bone
	-@${MPIEXEC} -n 1 ./ex6 -draw_cmap jet
	-@${MPIEXEC} -n 1 ./ex6 -draw_cmap coolwarm
	-@${MPIEXEC} -n 1 ./ex6 -draw_cmap parula
	-@${MPIEXEC} -n 1 ./ex6 -draw_cmap viridis
	-@${MPIEXEC} -n 1 ./ex6 -draw_cmap plasma
runex7:
	-@${MPIEXEC} -n 1 ./ex7
runex9:
	-@${MPIEXEC} -n 1 ./ex9
runex10:
	-@${MPIEXEC} -n 1 ./ex10
runex11:
	-@${MPIEXEC} -n 1 ./ex11
runex12:
	-@${MPIEXEC} -n 1 ./ex12

TESTEXAMPLES_C_X	  = ex1.PETSc runex1 ex1.rm ex2.PETSc runex2 ex2.rm ex3.PETSc runex3 ex3.rm \
                            ex4.PETSc runex4 runex4_db runex4_dv runex4_df ex4.rm \
                            ex6.PETSc runex6 ex6.rm ex7.PETSc runex7 ex7.rm ex9.PETSc runex9 ex9.rm \
                            ex10.PETSc runex10 ex10.rm ex11.PETSc runex11 ex11.rm ex12.PETSc runex12 ex12.rm
TESTEXAMPLES_FORTRAN	  = ex4f.PETSc ex4f.rm ex5f.PETSc ex5f.rm
TESTEXAMPLES_C_X_MPIUNI   = ex1.PETSc ex1.rm
TESTEXAMPLES_13		  = ex9.PETSc ex9.rm

include ${PETSC_DIR}/lib/petsc/conf/test
