#
CFLAGS		 =
FFLAGS		 =
CPPFLAGS         =
FPPFLAGS         =
MANSEC           = Vec
LOCDIR		 = src/vec/vec/examples/tutorials/
EXAMPLESC	 = ex1.c ex2.c ex3.c ex5.c ex6.c ex7.c ex8.c ex9.c ex10.c \
                ex11.c ex12.c ex15.c ex16.c  ex18.c ex19.c ex42.c ex42a.c ex21.c
EXAMPLESF	 = ex1f.F ex1f90.F90 ex2f.F ex3f.F ex4f.F ex4f90.F90 ex7f.F ex9f.F90 ex11f.F ex14f.F90 ex20f90.F90 ex21f90.F90

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

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

ex21f90: ex21f90.o ex21.o  chkopts
	-${FLINKER} -o ex21f90 ex21f90.o ex21.o ${PETSC_VEC_LIB}
	${RM} -f ex21f90.o ex21.o

ex42: ex42.o chkopts
	-${CLINKER} -o ex42 ex42.o ${PETSC_SNES_LIB}
	${RM} ex42.o

ex42a: ex42a.o chkopts
	-${CLINKER} -o ex42a ex42a.o ${PETSC_SNES_LIB}
	${RM} ex42a.o

ex43: ex43.o chkopts
	-${CLINKER} -o ex43 ex43.o ${PETSC_MIXED_LIB}
	${RM} ex43.o

include ${PETSC_DIR}/lib/petsc/conf/test
