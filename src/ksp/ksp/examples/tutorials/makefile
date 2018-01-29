CFLAGS	         =
FFLAGS	         =
CPPFLAGS         =
FPPFLAGS         =
LOCDIR           = src/ksp/ksp/examples/tutorials/
DIRS             = network
EXAMPLESC        = ex1.c ex2.c ex3.c ex4.c ex5.c ex6.c ex7.c ex8.c ex9.c \
                ex10.c ex11.c ex12.c ex13.c ex15.c ex16.c ex18.c ex23.c \
                ex25.c ex27.c ex28.c ex29.c ex30.c ex32.c ex34.c \
                ex41.c ex42.c ex43.c \
                ex45.c ex46.c  ex49.c ex50.c ex51.c ex52.c ex53.c \
                ex54.c ex55.c ex56.c ex58.c ex62.c ex63.cxx ex64.c ex65.c ex66.c ex67.c ex68.c ex69.c ex70.c ex100.c
EXAMPLESF        = ex1f.F90 ex2f.F ex6f.F ex11f.F ex13f90.F90 ex14f.F ex15f.F ex21f.F ex22f.F ex44f.F90 ex45f.F \
                   ex52f.F ex54f.F90 ex61f.F90 ex100f.F
MANSEC           = KSP
CLEANFILES       = rhs.vtk solution.vtk
NP               = 1

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex1: ex1.o  chkopts
	-${CLINKER} -o ex1 ex1.o  ${PETSC_KSP_LIB}
	${RM} ex1.o

ex1f: ex1f.o  chkopts
	-${FLINKER} -o ex1f ex1f.o  ${PETSC_KSP_LIB}
	${RM} ex1f.o

ex2: ex2.o  chkopts
	-${CLINKER} -o ex2 ex2.o  ${PETSC_KSP_LIB}
	${RM} ex2.o

ex2a: ex2a.o  chkopts
	-${CLINKER} -o ex2a ex2a.o  ${PETSC_KSP_LIB}
	${RM} ex2a.o

ex2f: ex2f.o  chkopts
	-${FLINKER} -o ex2f ex2f.o  ${PETSC_KSP_LIB}
	${RM} ex2f.o

ex3: ex3.o  chkopts
	-${CLINKER} -o ex3 ex3.o  ${PETSC_KSP_LIB}
	${RM} ex3.o

ex4: ex4.o  chkopts
	-${CLINKER} -o ex4 ex4.o  ${PETSC_KSP_LIB}
	${RM} ex4.o

ex5: ex5.o  chkopts
	-${CLINKER} -o ex5 ex5.o  ${PETSC_KSP_LIB}
	${RM} ex5.o

ex6: ex6.o  chkopts
	-${CLINKER} -o ex6 ex6.o  ${PETSC_KSP_LIB}
	${RM} ex6.o

ex6f: ex6f.o  chkopts
	-${FLINKER} -o ex6f ex6f.o  ${PETSC_KSP_LIB}
	${RM} ex6f.o

ex7: ex7.o  chkopts
	-${CLINKER} -o ex7 ex7.o  ${PETSC_KSP_LIB}
	${RM} ex7.o

ex8: ex8.o  chkopts
	-${CLINKER} -o ex8 ex8.o  ${PETSC_KSP_LIB}
	${RM} ex8.o

ex9: ex9.o  chkopts
	-${CLINKER} -o ex9 ex9.o  ${PETSC_KSP_LIB}
	${RM} ex9.o

ex10: ex10.o  chkopts
	-${CLINKER} -o ex10 ex10.o  ${PETSC_KSP_LIB}
	${RM} ex10.o

ex11: ex11.o  chkopts
	-${CLINKER} -o ex11 ex11.o  ${PETSC_KSP_LIB}
	${RM} ex11.o

ex11f: ex11f.o  chkopts
	-${FLINKER} -o ex11f ex11f.o  ${PETSC_KSP_LIB}
	${RM} ex11f.o

ex12: ex12.o  chkopts
	-${CLINKER} -o ex12 ex12.o  ${PETSC_KSP_LIB}
	${RM} ex12.o

ex13: ex13.o  chkopts
	-${CLINKER} -o ex13 ex13.o  ${PETSC_KSP_LIB}
	${RM} ex13.o

ex13f90: ex13f90.o  chkopts
	-${FLINKER} -o ex13f90 ex13f90.o  ${PETSC_KSP_LIB}
	${RM} ex13f90.o

ex14: ex14.o  chkopts
	-${CLINKER} -o ex14 ex14.o  ${PETSC_KSP_LIB}
	${RM} ex14.o

ex14f: ex14f.o  chkopts
	-${FLINKER} -o ex14f ex14f.o  ${PETSC_KSP_LIB}
	${RM} ex14f.o

ex15: ex15.o  chkopts
	-${CLINKER} -o ex15 ex15.o  ${PETSC_KSP_LIB}
	${RM} ex15.o

ex15f: ex15f.o  chkopts
	-${FLINKER} -o ex15f ex15f.o  ${PETSC_KSP_LIB}
	${RM} ex15f.o

ex16: ex16.o  chkopts
	-${CLINKER} -o ex16 ex16.o  ${PETSC_KSP_LIB}
	${RM} ex16.o

ex17: ex17.o  chkopts
	-${CLINKER} -o ex17 ex17.o  ${PETSC_KSP_LIB}
	${RM} ex17.o

ex18: ex18.o  chkopts
	-${CLINKER} -o ex18 ex18.o  ${PETSC_KSP_LIB}
	${RM} ex18.o

ex20: ex20.o  chkopts
	-${CLINKER} -o ex20 ex20.o  ${PETSC_KSP_LIB}
	${RM} ex20.o

ex21f: ex21f.o  chkopts
	-${FLINKER} -o ex21f ex21f.o  ${PETSC_KSP_LIB}
	${RM} ex21f.o

ex22: ex22.o  chkopts
	-${CLINKER} -o ex22 ex22.o  ${PETSC_SNES_LIB}
	${RM} ex22.o

ex22f: ex22f.o  chkopts
	-${FLINKER} -o ex22f ex22f.o  ${PETSC_SNES_LIB}
	${RM} ex22f.o

ex23: ex23.o  chkopts
	-${CLINKER} -o ex23 ex23.o  ${PETSC_KSP_LIB}
	${RM} ex23.o

ex25: ex25.o  chkopts
	-${CLINKER} -o ex25 ex25.o  ${PETSC_SNES_LIB}
	${RM} ex25.o

ex26: ex26.o  chkopts
	-${CLINKER} -o ex26 ex26.o  ${PETSC_KSP_LIB}
	${RM} ex26.o

ex27: ex27.o  chkopts
	-${CLINKER} -o ex27 ex27.o  ${PETSC_KSP_LIB}
	${RM} ex27.o

ex28: ex28.o  chkopts
	-${CLINKER} -o ex28 ex28.o  ${PETSC_SNES_LIB}
	${RM} ex28.o

ex29: ex29.o  chkopts
	-${CLINKER} -o ex29 ex29.o  ${PETSC_SNES_LIB}
	${RM} ex29.o

ex30: ex30.o  chkopts
	-${CLINKER} -o ex30 ex30.o  ${PETSC_KSP_LIB}
	${RM} ex30.o

ex32: ex32.o  chkopts
	-${CLINKER} -o ex32 ex32.o  ${PETSC_SNES_LIB}
	${RM} ex32.o

ex33: ex33.o chkopts
	-${CLINKER} -o ex33 ex33.o ${PETSC_SNES_LIB}
	${RM} ex33.o

ex34: ex34.o chkopts
	-${CLINKER} -o ex34 ex34.o ${PETSC_SNES_LIB}
	${RM} ex34.o

ex35: ex35.o chkopts
	-${CLINKER} -o ex35 ex35.o ${PETSC_SNES_LIB}
	${RM} ex35.o

ex36: ex36.o chkopts
	-${CLINKER} -o ex36 ex36.o ${PETSC_SNES_LIB}
	${RM} ex36.o

ex37: ex37.o chkopts
	-${CLINKER} -o ex37 ex37.o ${PETSC_SNES_LIB}
	${RM} ex37.o

ex38: ex38.o chkopts
	-${CLINKER} -o ex38 ex38.o ${PETSC_SNES_LIB}
	${RM} ex38.o

ex39: ex39.o chkopts
	-${CLINKER} -o ex39 ex39.o ${PETSC_SNES_LIB}
	${RM} ex39.o

ex40: ex40.o chkopts
	-${CLINKER} -o ex40 ex40.o ${PETSC_SNES_LIB}
	${RM} ex40.o

ex41: ex41.o chkopts
	-${CLINKER} -o ex41 ex41.o ${PETSC_SNES_LIB}
	${RM} ex41.o

ex42: ex42.o chkopts
	-${CLINKER} -o ex42 ex42.o ${PETSC_KSP_LIB}
	${RM} ex42.o

ex43: ex43.o chkopts
	-${CLINKER} -o ex43 ex43.o ${PETSC_KSP_LIB}
	${RM} ex43.o

ex44f: ex44f.o  chkopts
	-${FLINKER} -o ex44f ex44f.o  ${PETSC_KSP_LIB}
	${RM} ex44f.o

ex45f: ex45f.o  chkopts
	-${FLINKER} -o ex45f ex45f.o  ${PETSC_KSP_LIB}
	${RM} ex45f.o

ex45: ex45.o chkopts
	-${CLINKER} -o ex45 ex45.o ${PETSC_KSP_LIB}
	${RM} ex45.o

ex46: ex46.o chkopts
	-${CLINKER} -o ex46 ex46.o ${PETSC_KSP_LIB}
	${RM} ex46.o

ex47: ex47.o chkopts
	-${CLINKER} -o ex47 ex47.o ${PETSC_KSP_LIB}
	${RM} ex47.o

ex49: ex49.o chkopts
	-${CLINKER} -o ex49 ex49.o ${PETSC_KSP_LIB}
	${RM} ex49.o

ex50: ex50.o chkopts
	-${CLINKER} -o ex50 ex50.o ${PETSC_KSP_LIB}
	${RM} ex50.o

ex51: ex51.o chkopts
	-${CLINKER} -o ex51 ex51.o ${PETSC_KSP_LIB}
	${RM} ex51.o

ex52: ex52.o chkopts
	-${CLINKER} -o ex52 ex52.o ${PETSC_KSP_LIB}
	${RM} ex52.o

ex52f: ex52f.o  chkopts
	-${FLINKER} -o ex52f ex52f.o  ${PETSC_KSP_LIB}
	${RM} ex52f.o

ex53: ex53.o chkopts
	-${CLINKER} -o ex53 ex53.o ${PETSC_KSP_LIB}
	${RM} ex53.o

ex54: ex54.o chkopts
	-${CLINKER} -o ex54 ex54.o ${PETSC_KSP_LIB}
	${RM} ex54.o

ex54f: ex54f.o  chkopts
	-${FLINKER} -o ex54f ex54f.o  ${PETSC_KSP_LIB}
	${RM} ex54f.o

ex55: ex55.o chkopts
	-${CLINKER} -o ex55 ex55.o ${PETSC_KSP_LIB}
	${RM} ex55.o

ex56: ex56.o chkopts
	-${CLINKER} -o ex56 ex56.o ${PETSC_KSP_LIB}
	${RM} ex56.o

ex57f: ex57f.o  chkopts
	-${FLINKER} -o ex57f ex57f.o  ${PETSC_KSP_LIB}
	${RM} ex57f.o

ex58: ex58.o chkopts
	-${CLINKER} -o ex58 ex58.o ${PETSC_KSP_LIB}
	${RM} ex58.o

ex59: ex59.o chkopts
	-${CLINKER} -o ex59 ex59.o ${PETSC_KSP_LIB}
	${RM} ex59.o

ex60: ex60.o  chkopts
	-${CLINKER} -o ex60 ex60.o  ${PETSC_KSP_LIB}
	${RM} ex60.o

ex61f: ex61f.o  chkopts
	-${FLINKER} -o ex61f ex61f.o  ${PETSC_KSP_LIB}
	${RM} ex61f.o

ex62: ex62.o  chkopts
	-${CLINKER} -o ex62 ex62.o  ${PETSC_KSP_LIB}
	${RM} ex62.o

ex63: ex63.o  chkopts
	-${CLINKER} -o ex63 ex63.o  ${PETSC_KSP_LIB}
	${RM} ex63.o

ex64: ex64.o  chkopts
	-${CLINKER} -o ex64 ex64.o  ${PETSC_KSP_LIB}
	${RM} ex64.

ex65: ex65.o  chkopts
	-${CLINKER} -o ex65 ex65.o  ${PETSC_KSP_LIB}
	${RM} ex65.o

ex66: ex66.o  chkopts
	-${CLINKER} -o ex66 ex66.o  ${PETSC_KSP_LIB}
	${RM} ex66.o

ex67: ex67.o  chkopts
	-${CLINKER} -o ex67 ex67.o  ${PETSC_KSP_LIB}
	${RM} ex67.o

ex68: ex68.o  chkopts
	-${CLINKER} -o ex68 ex68.o  ${PETSC_KSP_LIB}
	${RM} ex68.o

ex69: ex69.o  chkopts
	-${CLINKER} -o ex69 ex69.o  ${PETSC_KSP_LIB}
	${RM} ex69.o

ex70: ex70.o  chkopts
	-${CLINKER} -o ex70 ex70.o  ${PETSC_KSP_LIB}
	${RM} ex70.o

ex100: ex100.o  chkopts
	-${CLINKER} -o ex100 ex100.o  ${PETSC_KSP_LIB}
	${RM} ex100.o

ex100f: ex100f.o  chkopts
	-${FLINKER} -o ex100f ex100f.o  ${PETSC_KSP_LIB}
	${RM} ex100f.o

testex100: ex100.PETSc
	-@if [ "${PETSC_WITH_BATCH}" != "" ]; then \
           echo "Running with batch filesystem; to test run src/ksp/ksp/examples/tutorials/ex100 with" ; \
           echo "your systems batch system"; \
        elif [ "${MPIEXEC}" = "/bin/false" ]; then \
           echo "*mpiexec not found*. Please run src/ksp/ksp/examples/tutorials/ex100 manually"; \
	elif [ -f ex100 ]; then \
           ${MPIEXEC} -n 1 ./ex100 -test  > ex100_1.tmp 2>&1; \
	   if (${DIFF} output/ex100_1.testout ex100_1.tmp > /dev/null 2>&1) then \
           echo "C/C++ example src/ksp/ksp/examples/tutorials/ex100 run successfully with 1 MPI process"; \
	   else echo "Possible error running C/C++ src/ksp/ksp/examples/tutorials/ex100 with 1 MPI process"; \
           echo "See http://www.mcs.anl.gov/petsc/documentation/faq.html";\
           cat ex100_1.tmp; fi; \
        ${RM} -f ex100_1.tmp; fi

include ${PETSC_DIR}/lib/petsc/conf/test
