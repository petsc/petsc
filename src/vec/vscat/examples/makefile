
CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/vec/vscat/examples/tests/
EXAMPLESC       = ex1.c
EXAMPLESF       =
MANSEC          = Vec

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex1: ex1.o  chkopts
	-${CLINKER} -o ex1 ex1.o ${PETSC_VEC_LIB}
	${RM} -f ex1.o

ex2: ex2.o  chkopts
	-${CLINKER} -o ex2 ex2.o ${PETSC_VEC_LIB}
	${RM} -f ex2.o

ex3: ex3.o  chkopts
	-${CLINKER} -o ex3 ex3.o ${PETSC_VEC_LIB}
	${RM} -f ex3.o
#--------------------------------------------------------------------------
runex1:
	-@${MPIEXEC} -n 2 ./ex1 -vecscatter_type mpi3node > ex1_1.tmp 2>&1;\
	   if (${DIFF} output/ex1_1.out ex1_1.tmp) then true ;  \
	   else printf "${PWD}\nPossible problem with ex1_1, diffs above\n=========================================\n"; fi ;\
	   ${RM} -f ex1_1.tmp
runex1_2:
	-@${MPIEXEC} -n 4 ./ex1 -vecscatter_type mpi3node > ex1_2.tmp 2>&1;\
	   if (${DIFF} output/ex1_2.out ex1_2.tmp) then true ;  \
	   else printf "${PWD}\nPossible problem with ex1_2, diffs above\n=========================================\n"; fi ;\
	   ${RM} -f ex1_2.tmp
runex1_3:
	-@${MPIEXEC} -n 2 ./ex1 -bs 2 -vecscatter_type mpi3node > ex1_1.tmp 2>&1;\
	   if (${DIFF} output/ex1_3.out ex1_1.tmp) then true ;  \
	   else printf "${PWD}\nPossible problem with ex1_3, diffs above\n=========================================\n"; fi ;\
	   ${RM} -f ex1_1.tmp

runex2:
	-@${MPIEXEC} -n 2 ./ex2 -vecscatter_type mpi3node > ex2_1.tmp 2>&1;\
           if (${DIFF} output/ex2_1.out ex2_1.tmp) then true ;  \
           else printf "${PWD}\nPossible problem with ex2_1, diffs above\n=========================================\n"; fi ;\
           ${RM} -f ex2_1.tmp
runex2_2:
	-@${MPIEXEC} -n 2 ./ex2 -vecscatter_type mpi3 > ex2_1.tmp 2>&1;\
           if (${DIFF} output/ex2_1.out ex2_1.tmp) then true ;  \
           else printf "${PWD}\nPossible problem with ex2_2, diffs above\n=========================================\n"; fi ;\
           ${RM} -f ex2_1.tmp
runex2_3:
	-@${MPIEXEC} -n 2 ./ex2 -bs 3 -vecscatter_type mpi3node > ex2_1.tmp 2>&1;\
           if (${DIFF} output/ex2_2.out ex2_1.tmp) then true ;  \
           else printf "${PWD}\nPossible problem with ex2_3, diffs above\n=========================================\n"; fi ;\
           ${RM} -f ex2_1.tmp
runex2_4:
	-@${MPIEXEC} -n 2 ./ex2 -bs 3 -vecscatter_type mpi3 > ex2_1.tmp 2>&1;\
           if (${DIFF} output/ex2_2.out ex2_1.tmp) then true ;  \
           else printf "${PWD}\nPossible problem with ex2_4, diffs above\n=========================================\n"; fi ;\
           ${RM} -f ex2_1.tmp
runex2_5:
	-@${MPIEXEC} -n 3 ./ex2 -vecscatter_type mpi3node > ex2_1.tmp 2>&1;\
           if (${DIFF} output/ex2_5.out ex2_1.tmp) then true ;  \
           else printf "${PWD}\nPossible problem with ex2_5, diffs above\n=========================================\n"; fi ;\
           ${RM} -f ex2_1.tmp
runex2_6:
	-@${MPIEXEC} -n 3 ./ex2 -vecscatter_type mpi3 > ex2_1.tmp 2>&1;\
           if (${DIFF} output/ex2_5.out ex2_1.tmp) then true ;  \
           else printf "${PWD}\nPossible problem with ex2_6, diffs above\n=========================================\n"; fi ;\
           ${RM} -f ex2_1.tmp

runex3:
	-@${MPIEXEC} -n 2 ./ex3 -vecscatter_type mpi3node > ex3_1.tmp 2>&1;\
	   if (${DIFF} output/ex3_1.out ex3_1.tmp) then true ;  \
	   else printf "${PWD}\nPossible problem with ex3_1, diffs above\n=========================================\n"; fi ;\
	   ${RM} -f ex3_1.tmp
runex3_2:
	-@${MPIEXEC} -n 2 ./ex3 -vecscatter_type mpi3 > ex3_1.tmp 2>&1;\
	   if (${DIFF} output/ex3_1.out ex3_1.tmp) then true ;  \
	   else printf "${PWD}\nPossible problem with ex3_2, diffs above\n=========================================\n"; fi ;\
	   ${RM} -f ex3_1.tmp
runex3_3:
	-@${MPIEXEC} -n 2 ./ex3 -bs 2 -vecscatter_type mpi3node > ex3_1.tmp 2>&1;\
	   if (${DIFF} output/ex3_3.out ex3_1.tmp) then true ;  \
	   else printf "${PWD}\nPossible problem with ex3_3, diffs above\n=========================================\n"; fi ;\
	   ${RM} -f ex3_1.tmp
runex3_4:
	-@${MPIEXEC} -n 2 ./ex3 -bs 2 -vecscatter_type mpi3 > ex3_1.tmp 2>&1;\
	   if (${DIFF} output/ex3_3.out ex3_1.tmp) then true ;  \
	   else printf "${PWD}\nPossible problem with ex3_4, diffs above\n=========================================\n"; fi ;\
	   ${RM} -f ex3_1.tmp
runex3_5:
	-@${MPIEXEC} -n 3 ./ex3 -vecscatter_type mpi3node > ex3_1.tmp 2>&1;\
	   if (${DIFF} output/ex3_5.out ex3_1.tmp) then true ;  \
	   else printf "${PWD}\nPossible problem with ex3_5, diffs above\n=========================================\n"; fi ;\
	   ${RM} -f ex3_1.tmp
runex3_6:
	-@${MPIEXEC} -n 3 ./ex3 -vecscatter_type mpi3 > ex3_1.tmp 2>&1;\
	   if (${DIFF} output/ex3_5.out ex3_1.tmp) then true ;  \
	   else printf "${PWD}\nPossible problem with ex3_6, diffs above\n=========================================\n"; fi ;\
	   ${RM} -f ex3_1.tmp


TESTEXAMPLES_C		    = ex1.PETSc runex1 runex1_2 runex1_3 ex1.rm \
                              ex2.PETSc runex2 runex2_2 runex2_3 runex2_4 ex2.rm \
                              ex3.PETSc runex3 runex3_2 runex3_3 runex3_4 ex3.rm

include ${PETSC_DIR}/lib/petsc/conf/test
