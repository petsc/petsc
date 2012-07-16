#
CFLAGS		 = 
FFLAGS		 = 
CPPFLAGS         =
FPPFLAGS         =
MANSEC           = Vec
LOCDIR		 = src/vec/vec/examples/tutorials/
EXAMPLESC	 = ex1.c ex2.c ex3.c ex5.c ex6.c ex7.c ex8.c ex9.c ex10.c \
                ex11.c ex12.c ex15.c ex16.c  ex18.c ex19.c ex42.c ex42a.c
EXAMPLESF	 = ex1f.F ex1f90.F ex2f.F ex3f.F ex4f.F ex4f90.F ex7f.F ex9f.F ex11f.F ex14f.F ex20f90.F90

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

ex1: ex1.o  chkopts 
	-${CLINKER} -o ex1 ex1.o ${PETSC_VEC_LIB}
	${RM} -f ex1.o

ex1f: ex1f.o  chkopts
	-${FLINKER} -o ex1f ex1f.o ${PETSC_VEC_LIB}
#	${RM} -f ex1f.o

ex1f90: ex1f90.o  chkopts
	-${FLINKER} -o ex1f90 ex1f90.o ${PETSC_VEC_LIB}
	${RM} -f ex1f90.o

ex2: ex2.o  chkopts
	-${CLINKER} -o ex2 ex2.o  ${PETSC_VEC_LIB}
	${RM} -f ex2.o

ex2f: ex2f.o  chkopts
	-${FLINKER} -o ex2f ex2f.o ${PETSC_VEC_LIB}
	${RM} -f ex2f.o

ex3: ex3.o  chkopts
	-${CLINKER} -o ex3 ex3.o  ${PETSC_VEC_LIB}
	${RM} -f ex3.o

ex3f: ex3f.o  chkopts
	-${FLINKER} -o ex3f ex3f.o ${PETSC_VEC_LIB}
	${RM} -f ex3f.o

ex4: ex4.o  chkopts
	-${CLINKER} -o ex4 ex4.o  ${PETSC_VEC_LIB}
	${RM} -f ex4.o

ex4f: ex4f.o  chkopts
	-${FLINKER} -o ex4f ex4f.o ${PETSC_VEC_LIB}
	${RM} -f ex4f.o

ex4f90: ex4f90.o  chkopts
	-${FLINKER} -o ex4f90 ex4f90.o ${PETSC_VEC_LIB}
	${RM} -f ex4f90.o

ex5: ex5.o  chkopts
	-${CLINKER} -o ex5 ex5.o  ${PETSC_VEC_LIB}
	${RM} -f ex5.o

ex6: ex6.o  chkopts
	-${CLINKER} -o ex6 ex6.o  ${PETSC_VEC_LIB}
	${RM} -f ex6.o

ex7: ex7.o ex7f.o chkopts
	-${CLINKER} -o ex7 ex7.o ex7f.o ${PETSC_VEC_LIB} 
	${RM} -f ex7.o ex7f.o

ex8: ex8.o  chkopts
	-${CLINKER} -o ex8 ex8.o  ${PETSC_VEC_LIB}
	${RM} -f ex8.o

ex9: ex9.o  chkopts
	-${CLINKER} -o ex9 ex9.o  ${PETSC_VEC_LIB}
	${RM} -f ex9.o

ex9f: ex9f.o  chkopts
	-${FLINKER} -o ex9f ex9f.o ${PETSC_VEC_LIB}
	${RM} -f ex9f.o

ex10: ex10.o  chkopts
	-${CLINKER} -o ex10 ex10.o  ${PETSC_VEC_LIB}
	${RM} -f ex10.o

ex11: ex11.o  chkopts
	-${CLINKER} -o ex11 ex11.o  ${PETSC_VEC_LIB}
	${RM} -f ex11.o

ex11f: ex11f.o  chkopts
	-${FLINKER} -o ex11f ex11f.o ${PETSC_VEC_LIB}
	${RM} -f ex11f.o

ex12: ex12.o  chkopts
	-${CLINKER} -o ex12 ex12.o  ${PETSC_VEC_LIB}
	${RM} -f ex12.o

ex13: ex13.o  chkopts
	-${CLINKER} -o ex13 ex13.o  ${PETSC_VEC_LIB}
	${RM} -f ex13.o

ex14f: ex14f.o  chkopts
	-${FLINKER} -o ex14f ex14f.o ${PETSC_VEC_LIB}
	${RM} -f ex14f.o

ex15: ex15.o  chkopts
	-${CLINKER} -o ex15 ex15.o  ${PETSC_VEC_LIB}
	${RM} -f ex15.o

ex16: ex16.o  chkopts
	-${CLINKER} -o ex16 ex16.o  ${PETSC_VEC_LIB}
	${RM} -f ex16.o

ex17: ex17.o  chkopts
	-${CLINKER} -o ex17 ex17.o  ${PETSC_VEC_LIB}
	${RM} -f ex17.o

ex18: ex18.o  chkopts
	-${CLINKER} -o ex18 ex18.o  ${PETSC_VEC_LIB}
	${RM} -f ex18.o

ex19: ex19.o  chkopts
	-${CLINKER} -o ex19 ex19.o  ${PETSC_VEC_LIB}
	${RM} -f ex19.o

ex20f90: ex20f90.o  chkopts
	-${FLINKER} -o ex20f90 ex20f90.o ${PETSC_VEC_LIB}
	${RM} -f ex20f90.o

ex42: ex42.o chkopts
	-${CLINKER} -o ex42 ex42.o ${PETSC_SNES_LIB}
	${RM} ex42.o

ex42a: ex42a.o chkopts
	-${CLINKER} -o ex42a ex42a.o ${PETSC_SNES_LIB}
	${RM} ex42a.o

ex43: ex43.o chkopts
	-${CLINKER} -o ex43 ex43.o ${PETSC_MIXED_LIB}
	${RM} ex43.o

#-------------------------------------------------------------------------------------------
#   This section is used for testing examples and can be ignored 
#-------------------------------------------------------------------------------------------
runex1:
	-@${MPIEXEC} -n 1 ./ex1 > ex1_1.tmp 2>&1;   \
	   if (${DIFF} output/ex1_1.out ex1_1.tmp) then true; \
	   else echo ${PWD} ; echo "Possible problem with with ex1_1, diffs above \n========================================="; fi; \
	   ${RM} -f ex1_1.tmp
runex1_2:
	-@${MPIEXEC}  -n 2 ./ex1 > ex1_2.tmp 2>&1;\
	   if (${DIFF} output/ex1_1.out ex1_2.tmp) then true ;  \
	   else echo ${PWD} ; echo "Possible problem with with ex1_2, diffs above \n========================================="; fi ;\
	   ${RM} -f ex1_2.tmp

runex1f:
	-@${MPIEXEC} -n 1  ./ex1f > ex1f_1.tmp 2>&1;   \
	   if (${DIFF} output/ex1f_1.out ex1f_1.tmp) then true; \
	   else echo ${PWD} ; echo "Possible problem with with ex1f_1, diffs above \n========================================="; fi; \
	   ${RM} -f ex1f_1.tmp
runex1f_2:
	-@${MPIEXEC}  -n 2 ./ex1f > ex1f_2.tmp 2>&1;\
	   if (${DIFF} output/ex1f_2.out ex1f_2.tmp) then true ;  \
	   else echo ${PWD} ; echo "Possible problem with with ex1f_2, diffs above \n========================================="; fi ;\
	   ${RM} -f ex1f_2.tmp
runex1f90:
	-@${MPIEXEC}  -n 2 ./ex1f90 > ex1f90_1.tmp 2>&1;\
	   if (${DIFF} output/ex1f90_1.out ex1f90_1.tmp) then true ;  \
	   else echo ${PWD} ; echo "Possible problem with with ex1f90_1, diffs above \n========================================="; fi ;\
	   ${RM} -f ex1f90_1.tmp
runex2:
	-@${MPIEXEC} -n 2 ./ex2 > ex2_1.tmp 2>&1;\
	   if (${DIFF} output/ex2_1.out ex2_1.tmp) then true; \
	   else echo ${PWD} ; echo "Possible problem with with ex2_1, diffs above \n========================================="; fi;\
	   ${RM} -f ex2_1.tmp
runex2f:
	-@${MPIEXEC}  -n 2 ./ex2f | grep -v "MPI processes" > ex2f_1.tmp 2>&1;\
	   if (${DIFF} output/ex2f_1.out ex2f_1.tmp) then true; \
	   else echo ${PWD} ; echo "Possible problem with with ex2f_1, diffs above \n========================================="; fi;\
	   ${RM} -f ex2f_1.tmp
runex3:
	-@${MPIEXEC}  -n 2 ./ex3 -nox > ex3_1.tmp 2>&1;\
	   if (${DIFF} output/ex3_1.out ex3_1.tmp) then true; \
	   else echo ${PWD} ; echo "Possible problem with with ex3_1, diffs above \n========================================="; fi;\
	   ${RM} -f ex3_1.tmp
runex3f:
	-@${MPIEXEC}  -n 2 ./ex3f > ex3f_1.tmp 2>&1;\
	   if (${DIFF} output/ex3f_1.out ex3f_1.tmp) then true; \
	   else echo ${PWD} ; echo "Possible problem with with ex3f_1, diffs above \n========================================="; fi;\
	   ${RM} -f ex3f_1.tmp
runex4f:
	-@${MPIEXEC} -n 1 ./ex4f > ex4f_1.tmp 2>&1;\
	   if (${DIFF} output/ex4f_1.out ex4f_1.tmp) then true; \
	   else echo ${PWD} ; echo "Possible problem with with ex4f_1, diffs above \n========================================="; fi;\
	   ${RM} -f ex4f_1.tmp
runex4f90:
	-@${MPIEXEC} -n 1 ./ex4f90 > ex4f90_1.tmp 2>&1;\
	   if (${DIFF} output/ex4f90_1.out ex4f90_1.tmp) then true; \
	   else echo ${PWD} ; echo "Possible problem with with ex4f90_1, diffs above \n========================================="; fi;\
	   ${RM} -f ex4f90_1.tmp
runex5:
	-@${RM} -f vector.dat.info vector.dat
	-@${MPIEXEC} -n 1 ./ex5 > ex5_1.tmp 2>&1;\
	   if (${DIFF} output/ex5_1.out ex5_1.tmp) then true; \
	   else echo ${PWD} ; echo "Possible problem with with ex5_1, diffs above \n========================================="; fi
	-@${RM} -f ex5_1.tmp vector.dat vector.dat.info
runex5_2:
	-@${RM} -f vector.dat.info vector.dat
	-@${MPIEXEC} -n 2 ./ex5 > ex5_2.tmp 2>&1;\
	   if (${DIFF} output/ex5_2.out ex5_2.tmp) then true; \
	   else echo ${PWD} ; echo "Possible problem with with ex5_2, diffs above \n========================================="; fi
	-@${RM} -f ex5_2.tmp vector.dat vector.dat.info
runex6:
	-@${RM} -f input.dat input.dat.info
	-@${MPIEXEC} -n 1 ./ex6 > ex6_1.tmp 2>&1;\
	   if (${DIFF} output/ex6_1.out ex6_1.tmp) then true; \
	   else echo ${PWD} ; echo "Possible problem with with ex6_1, diffs above \n========================================="; fi
	-@${RM} -f ex6_1.tmp input.dat input.dat.info
runex7:
	-@${RM} -f input.dat input.dat.info
	-@${MPIEXEC} -n 3 ./ex7 | sort |grep -v "MPI processes" > ex7_1.tmp 2>&1;\
	   if (${DIFF} output/ex7_1.out ex7_1.tmp) then true; \
	   else echo ${PWD} ; echo "Possible problem with with ex7_1, diffs above \n========================================="; fi
	-@${RM} -f ex7_1.tmp input.dat input.dat.info
runex8:
	-@${MPIEXEC} -n 4 ./ex8 > ex8_1.tmp 2>&1;\
	   if (${DIFF} output/ex8_1.out ex8_1.tmp) then true; \
	   else echo ${PWD} ; echo "Possible problem with with ex8_1, diffs above \n========================================="; fi;\
	   ${RM} -f ex8_1.tmp
runex9:
	-@${MPIEXEC} -n 2 ./ex9 > ex9_1.tmp 2>&1;\
	   if (${DIFF} output/ex9_1.out ex9_1.tmp) then true; \
	   else echo ${PWD} ; echo "Possible problem with with ex9_1, diffs above \n========================================="; fi;\
	   ${RM} -f ex9_1.tmp
runex9_2:
	-@${MPIEXEC} -n 2 ./ex9 -allocate > ex9_2.tmp 2>&1;\
	   if (${DIFF} output/ex9_1.out ex9_2.tmp) then true; \
	   else echo ${PWD} ; echo "Possible problem with with ex9_2, diffs above \n========================================="; fi;\
	   ${RM} -f ex9_2.tmp
runex9_3:
	-@${MPIEXEC} -n 2 ./ex9 -vecmpisetghost > ex9_3.tmp 2>&1;\
	   if (${DIFF} output/ex9_1.out ex9_3.tmp) then true; \
	   else echo ${PWD} ; echo "Possible problem with with ex9_3, diffs above \n========================================="; fi;\
	   ${RM} -f ex9_3.tmp
runex9f:
	-@${MPIEXEC} -n 2 ./ex9f > ex9f_1.tmp 2>&1;\
	   if (${DIFF} output/ex9f_1.out ex9f_1.tmp) then true; \
	   else echo ${PWD} ; echo "Possible problem with with ex9f_1, diffs above \n========================================="; fi;\
	   ${RM} -f ex9f_1.tmp
runex9f_2:
	-@${MPIEXEC} -n 2 ./ex9f -allocate > ex9f_1.tmp 2>&1;\
	   if (${DIFF} output/ex9f_1.out ex9f_1.tmp) then true; \
	   else echo ${PWD} ; echo "Possible problem with with ex9f_2, diffs above \n========================================="; fi;\
	   ${RM} -f ex9f_1.tmp

runex10:
	-@${MPIEXEC} -n 2 ./ex10 -binary > ex10_1.tmp 2>&1;\
	   if (${DIFF} output/ex10_1.out ex10_1.tmp) then true; \
	   else echo ${PWD} ; echo "Possible problem with with ex10_1, diffs above \n========================================="; fi;\
	   ${RM} -f ex10_1.tmp
runex10_2:
	-@${MPIEXEC} -n 3 ./ex10 -binary > ex10_2.tmp 2>&1;\
	   if (${DIFF} output/ex10_2.out ex10_2.tmp) then true; \
	   else echo ${PWD} ; echo "Possible problem with with ex10_2, diffs above \n========================================="; fi;\
	   ${RM} -f ex10_2.tmp
runex10_3:
	-@${MPIEXEC} -n 5 ./ex10 -binary > ex10_3.tmp 2>&1;\
	   if (${DIFF} output/ex10_3.out ex10_3.tmp) then true; \
	   else echo ${PWD} ; echo "Possible problem with with ex10_3, diffs above \n========================================="; fi;\
	   ${RM} -f ex10_3.tmp

runex11:
	-@${MPIEXEC}  -n 2 ./ex11 > ex11_1.tmp 2>&1;\
	   if (${DIFF} output/ex11_1.out ex11_1.tmp) then true ;  \
	   else echo ${PWD} ; echo "Possible problem with with ex11_1, diffs above \n========================================="; fi ;\
	   ${RM} -f ex11_1.tmp
runex11f:
	-@${MPIEXEC}  -n 2 ./ex11f > ex11f_1.tmp 2>&1;\
	   if (${DIFF} output/ex11f_1.out ex11f_1.tmp) then true ;  \
	   else echo ${PWD} ; echo "Possible problem with with ex11f_1, diffs above \n========================================="; fi ;\
	   ${RM} -f ex11f_1.tmp
runex12:
	-@${MPIEXEC}  -n 2 ./ex12 > ex12_1.tmp 2>&1;\
	   if (${DIFF} output/ex12_1.out ex12_1.tmp) then true ;  \
	   else echo ${PWD} ; echo "Possible problem with with ex12_1, diffs above \n========================================="; fi ;\
	   ${RM} -f ex12_1.tmp
runex16:
	-@${MPIEXEC}  -n 2 ./ex16 > ex16_1.tmp 2>&1;\
	   if (${DIFF} output/ex16_1.out ex16_1.tmp) then true ;  \
	   else echo ${PWD} ; echo "Possible problem with with ex16_1, diffs above \n========================================="; fi ;\
	   ${RM} -f ex16_1.tmp
runex20f90:
	-@${MPIEXEC} -n 1 ./ex20f90 > ex20f90_1.tmp 2>&1;\
	   if (${DIFF} output/ex20f90_1.out ex20f90_1.tmp) then true; \
	   else echo ${PWD} ; echo "Possible problem with with ex20f90_1, diffs above \n========================================="; fi;\
	   ${RM} -f ex20f90_1.tmp

TESTEXAMPLES_C		    = ex1.PETSc runex1 runex1_2 ex1.rm  \
                              ex2.PETSc runex2 ex2.rm ex5.PETSc runex5 runex5_2 ex5.rm ex6.PETSc \
                              runex6 ex6.rm ex8.PETSc runex8 ex8.rm ex10.PETSc runex10 runex10_2 runex10_3 ex10.rm ex11.PETSc runex11 ex11.rm \
                              ex12.PETSc runex12 ex12.rm ex16.PETSc runex16 ex16.rm ex9.PETSc runex9 runex9_2 ex9.rm ex16.PETSc \
                              ex16.rm 
TESTEXAMPLES_C_X	    = ex3.PETSc runex3 ex3.rm
TESTEXAMPLES_FORTRAN	    = ex1f.PETSc runex1f runex1f_2 ex1f.rm ex2f.PETSc runex2f ex2f.rm ex4f.PETSc \
                              runex4f ex4f.rm ex3f.PETSc ex3f.rm ex7.PETSc runex7 ex7.rm ex9f.PETSc runex9f \
                              runex9f_2 ex9f.rm ex11f.PETSc runex11f ex11f.rm ex14f.PETSc ex14f.rm
TESTEXAMPLES_C_X_MPIUNI   = ex1.PETSc runex1 ex1.rm ex5.PETSc runex5 ex5.rm
TESTEXAMPLES_FORTRAN_MPIUNI = ex1f.PETSc runex1f ex1f.rm ex4f.PETSc runex4f ex4f.rm ex7.PETSc ex7.rm
TESTEXAMPLES_F90	    = ex1f90.PETSc runex1f90 ex1f90.rm ex20f90.PETSc runex20f90 ex20f90.rm ex4f90.PETSc runex4f90 ex4f90.rm
include ${PETSC_DIR}/conf/test
