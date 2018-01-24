
CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/dm/examples/tutorials/
EXAMPLESC       = ex1.c ex2.c ex3.c ex4.c ex5.c ex6.c ex7.c ex9.c ex10.c ex12.c ex15.c ex19.c ex20.c ex51.c swarm_ex1.c swarm_ex2.c swarm_ex3.c
EXAMPLESF       = ex11f90.F90 ex13f90aux.F90 ex13f90.F90
MANSEC          = DM
SUBMANSEC       = DMDA
CLEANFILES      = output_file output_file.info

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

swarm_ex1: swarm_ex1.o   chkopts
	-${CLINKER} -o swarm_ex1 swarm_ex1.o  ${PETSC_DM_LIB}
swarm_ex2: swarm_ex2.o   chkopts
	-${CLINKER} -o swarm_ex2 swarm_ex2.o  ${PETSC_DM_LIB}
swarm_ex3: swarm_ex3.o   chkopts
	-${CLINKER} -o swarm_ex3 swarm_ex3.o  ${PETSC_DM_LIB}

ex1: ex1.o   chkopts
	-${CLINKER} -o ex1 ex1.o  ${PETSC_DM_LIB}
	${RM} -f ex1.o

ex2: ex2.o   chkopts
	-${CLINKER} -o ex2 ex2.o  ${PETSC_DM_LIB}
	${RM} -f ex2.o

ex3: ex3.o   chkopts
	-${CLINKER} -o ex3 ex3.o  ${PETSC_DM_LIB}
	${RM} -f ex3.o

ex4: ex4.o   chkopts
	-${CLINKER} -o ex4 ex4.o  ${PETSC_DM_LIB}
	${RM} -f ex4.o

ex5: ex5.o   chkopts
	-${CLINKER} -o ex5 ex5.o  ${PETSC_DM_LIB}
	${RM} -f ex5.o

ex6: ex6.o   chkopts
	-${CLINKER} -o ex6 ex6.o  ${PETSC_DM_LIB}
	${RM} -f ex6.o

ex6f90: ex6f90.o   chkopts
	-${FLINKER} -o ex6f90 ex6f90.o  ${PETSC_DM_LIB}
	${RM} -f ex6f90.o

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

ex11f90: ex11f90.o   chkopts
	-${FLINKER} -o ex11f90 ex11f90.o  ${PETSC_DM_LIB}
	${RM} -f ex11f90.o

ex12: ex12.o   chkopts
	-${CLINKER} -o ex12 ex12.o  ${PETSC_DM_LIB}
	${RM} -f ex12.o

ex13f90: ex13f90aux.o ex13f90.o chkopts
	-${FLINKER} -o ex13f90 ex13f90.o ex13f90aux.o ${PETSC_DM_LIB}
	${RM} -f ex13f90.o ex13f90aux.o ex13f90aux ex13f90aux.mod

ex14: ex14.o   chkopts
	-${CLINKER} -o ex14 ex14.o  ${PETSC_DM_LIB}
	${RM} -f ex14.o

ex15: ex15.o   chkopts
	-${CLINKER} -o ex15 ex15.o  ${PETSC_DM_LIB}
	${RM} -f ex15.o

ex19: ex19.o   chkopts
	-${CLINKER} -o ex19 ex19.o  ${PETSC_DM_LIB}
	${RM} -f ex19.o

ex20: ex20.o   chkopts
	-${CLINKER} -o ex20 ex20.o  ${PETSC_DM_LIB}
	${RM} -f ex20.o

ex21: ex21.o   chkopts
	-${CLINKER} -o ex21 ex21.o  ${PETSC_DM_LIB}
	${RM} -f ex21.o

ex51: ex51.o  chkopts
	-${CLINKER} -o ex51 ex51.o ${PETSC_SNES_LIB}
	${RM} ex51.o

#-----------------------------------------------------------------
runex13f90:
	-@${MPIEXEC} -n 4 ./ex13f90  > ex13f90.tmp 2>&1;   \
	   if (${DIFF} output/ex13f90_1.out ex13f90.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex13f90, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex13f90.tmp



ex65dm: ex65dm.o  chkopts
	-${CLINKER} -o ex65dm ex65dm.o ${PETSC_SNES_LIB}
	${RM} ex65dm.o

include ${PETSC_DIR}/lib/petsc/conf/test
