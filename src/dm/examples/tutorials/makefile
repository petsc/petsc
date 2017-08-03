
CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/dm/examples/tutorials/
EXAMPLESC       = ex1.c ex2.c ex3.c ex4.c ex5.c ex6.c ex7.c ex8.c ex9.c ex10.c ex12.c ex15.c ex51.c ex65dm.c swarm_ex1.c
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

ex65dm: ex65dm.o  chkopts
	-${CLINKER} -o ex65dm ex65dm.o ${PETSC_SNES_LIB}
	${RM} ex65dm.o

#-----------------------------------------------------------------
runex1:
	-@${MPIEXEC} -n 2 ./ex1 -nox
runex2:
	-@${MPIEXEC} -n 3 ./ex2
runex3:
	-@${MPIEXEC} -n 3 ./ex3 -mat_view  > ex3_1.tmp 2>&1;   \
	   if (${DIFF} output/ex3_1.out ex3_1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex3_1, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex3_1.tmp
runex3_2:
	-@${MPIEXEC} -n 3 ./ex3 -mat_view -dim 2 > ex3_2.tmp 2>&1;   \
	   if (${DIFF} output/ex3_2.out ex3_2.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex3_2, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex3_2.tmp
runex3_3:
	-@${MPIEXEC} -n 3 ./ex3 -mat_view -dim 3 > ex3_3.tmp 2>&1;   \
	   if (${DIFF} output/ex3_3.out ex3_3.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex3_3, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex3_3.tmp
runex5:
	-@${MPIEXEC} -n 1 ./ex5  > ex5_1.tmp 2>&1;   \
	   if (${DIFF} output/ex5_1.out ex5_1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex5_1, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex5_1.tmp
runex5_2:
	-@${MPIEXEC} -n 2 ./ex5  > ex5_2.tmp 2>&1;   \
	   if (${DIFF} output/ex5_2.out ex5_2.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex5_2, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex5_2.tmp
runex6:
	-@${MPIEXEC} -n 1 ./ex6  > ex6_1.tmp 2>&1;   \
	   if (${DIFF} output/ex6_1.out ex6_1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex6, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex6_1.tmp
runex6_2:
	-@${MPIEXEC} -n 2 ./ex6  > ex6_2.tmp 2>&1;   \
	   if (${DIFF} output/ex6_2.out ex6_2.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex6_2, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex6_2.tmp
runex6_4:
	-@${MPIEXEC} -n 4 ./ex6  > ex6_4.tmp 2>&1;   \
	   if (${DIFF} output/ex6_4.out ex6_4.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex6_4, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex6_4.tmp
runex6_9:
	-@${MPIEXEC} -n 9 ./ex6  > ex6_9.tmp 2>&1;   \
	   if (${DIFF} output/ex6_9.out ex6_9.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex6_9, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex6_9.tmp
runex7:
	-@${MPIEXEC} -n 1 ./ex7  > ex7.tmp 2>&1;   \
	   if (${DIFF} output/ex7.out ex7.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex7, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex7.tmp output_file output_file.info
runex9:
	-@${MPIEXEC} -n 4 ./ex9
	-@${MPIEXEC} -n 4 ./ex9 -ndof 2
runex10:
	-@${MPIEXEC} -n 4 ./ex10
	-@${MPIEXEC} -n 4 ./ex10 -viewer_hdf5_base_dimension2
	-@${MPIEXEC} -n 4 ./ex10 -viewer_hdf5_sp_output

runex11f90:
	-@${MPIEXEC} -n 1 ./ex11f90 | ${GREP} -v "Vec Object" > ex11f90.tmp 2>&1;   \
	   if (${DIFF} output/ex11f90.out ex11f90.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex11f90, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex11f90.tmp
runex13f90:
	-@${MPIEXEC} -n 4 ./ex13f90  > ex13f90.tmp 2>&1;   \
	   if (${DIFF} output/ex13f90.out ex13f90.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex13f90, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex13f90.tmp
runex15:
	-@${MPIEXEC} -n 1 ./ex15 > ex15.tmp 2>&1;   \
	   if (${DIFF} output/ex15_1.out ex15.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex15_1, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex15.tmp dmda.pbvec dmda.pbvec.info
runex15_2:
	-@${MPIEXEC} -n 12 ./ex15 > ex15.tmp 2>&1;   \
	   if (${DIFF} output/ex15_2.out ex15.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex15_2, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex15.tmp dmda.pbvec dmda.pbvec.info


TESTEXAMPLES_C		  = ex3.PETSc runex3 runex3_2 runex3_3 ex3.rm ex4.PETSc ex4.rm \
                            ex6.PETSc runex6 runex6_2 runex6_4 runex6_9 ex6.rm \
                            ex7.PETSc runex7 ex7.rm ex12.PETSc ex12.rm ex51.PETSc ex51.rm \
                            ex15.PETSc runex15 runex15_2 ex15.rm
TESTEXAMPLES_C_X	  = ex1.PETSc runex1 ex1.rm ex5.PETSc runex5 runex5_2 ex5.rm
TESTEXAMPLES_FORTRAN	  =
TESTEXAMPLES_F90_NOCOMPLEX= ex11f90.PETSc runex11f90 ex11f90.rm ex13f90.PETSc runex13f90 ex13f90.rm
TESTEXAMPLES_C_X_MPIUNI   =
TESTEXAMPLES_C_COMPLEX	  =
TESTEXAMPLES_C_NOCOMPLEX  = ex2.PETSc runex2 ex2.rm ex65dm.PETSc ex65dm.rm
TESTEXAMPLES_13		  =
TESTEXAMPLES_HDF5         = ex9.PETSc runex9 ex9.rm ex10.PETSc runex10 ex10.rm

include ${PETSC_DIR}/lib/petsc/conf/test
