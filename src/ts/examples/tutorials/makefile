
CFLAGS          =
FFLAGS          =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/ts/examples/tutorials/
EXAMPLESC       = ex1.c ex2.c ex3.c ex4.c ex5.c ex6.c ex7.c ex8.c \
                ex9.c ex10.c ex11.c ex12.c ex13.c ex14.c ex15.c ex16.c ex17.c \
                ex19.c ex20.c ex21.c ex22.c ex24.c ex25.c ex26.c \
                ex28.c ex31.c ex34.c ex35.cxx extchem.c\
                ex16adj.c ex16opt_p.c ex16opt_ic.c  \
                ex20adj.c ex20opt_p.c ex20opt_ic.c  \
                ex40.c ex41.c ex42.c ex48.c ex49.c
EXAMPLESF       = ex1f.F ex22f.F ex22f_mf.F90
MANSEC          = TS
DIRS            = phasefield advection-diffusion-reaction eimex power_grid network
CLEANFILES      = SA-data/*
include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex1: ex1.o  chkopts
	-${CLINKER} -o ex1 ex1.o  ${PETSC_TS_LIB}
	${RM} ex1.o

ex1f: ex1f.o  chkopts
	-${FLINKER} -o ex1f ex1f.o  ${PETSC_TS_LIB}
	${RM} ex1f.o

ex2: ex2.o chkopts
	-${CLINKER} -o ex2 ex2.o ${PETSC_TS_LIB}
	${RM} ex2.o

ex3: ex3.o  chkopts
	-${CLINKER} -o ex3 ex3.o  ${PETSC_TS_LIB}
	${RM} ex3.o

ex4: ex4.o  chkopts
	-${CLINKER} -o ex4 ex4.o  ${PETSC_TS_LIB}
	${RM} ex4.o

ex5: ex5.o  chkopts
	-${CLINKER} -o ex5 ex5.o  ${PETSC_TS_LIB}
	${RM} ex5.o

ex6: ex6.o  chkopts
	-${CLINKER} -o ex6 ex6.o  ${PETSC_TS_LIB}
	${RM} ex6.o

ex7: ex7.o  chkopts
	-${CLINKER} -o ex7 ex7.o  ${PETSC_TS_LIB}
	${RM} ex7.o

ex8: ex8.o  chkopts
	-${CLINKER} -o ex8 ex8.o  ${PETSC_TS_LIB}
	${RM} ex8.o

ex9: ex9.o  chkopts
	-${CLINKER} -o ex9 ex9.o  ${PETSC_TS_LIB}
	${RM} ex9.o

ex10: ex10.o  chkopts
	-${CLINKER} -o ex10 ex10.o  ${PETSC_TS_LIB}
	${RM} ex10.o

ex11: ex11.o  chkopts
	-${CLINKER} -o ex11 ex11.o  ${PETSC_TS_LIB}
	${RM} ex11.o

ex11_sa: ex11_sa.o chkopts
	-${CLINKER} -o ex11_sa ex11_sa.o ${PETSC_TS_LIB}
	${RM} ex11_sa.o

ex12: ex12.o  chkopts
	-${CLINKER} -o ex12 ex12.o  ${PETSC_TS_LIB}
	${RM} ex12.o

ex13: ex13.o  chkopts
	-${CLINKER} -o ex13 ex13.o  ${PETSC_TS_LIB}
	${RM} ex13.o

ex14: ex14.o  chkopts
	-${CLINKER} -o ex14 ex14.o  ${PETSC_TS_LIB}
	${RM} ex14.o

ex15: ex15.o  chkopts
	-${CLINKER} -o ex15 ex15.o  ${PETSC_TS_LIB}
	${RM} ex15.o

ex16: ex16.o  chkopts
	-${CLINKER} -o ex16 ex16.o  ${PETSC_TS_LIB}
	${RM} ex16.o

ex16adj: ex16adj.o chkopts
	-${CLINKER} -o ex16adj ex16adj.o ${PETSC_TS_LIB}
	${RM} ex16adj.o

ex16opt_p: ex16opt_p.o chkopts
	-${CLINKER} -o ex16opt_p ex16opt_p.o ${PETSC_LIB}
	${RM} ex16opt_p.o

ex16opt_ic: ex16opt_ic.o chkopts
	-${CLINKER} -o ex16opt_ic ex16opt_ic.o ${PETSC_LIB}
	${RM} ex16opt_ic.o

ex17: ex17.o  chkopts
	-${CLINKER} -o ex17 ex17.o  ${PETSC_TS_LIB}
	${RM} ex17.o

ex18: ex18.o  chkopts
	-${CLINKER} -o ex18 ex18.o  ${PETSC_TS_LIB}
	${RM} ex18.o

ex19: ex19.o  chkopts
	-${CLINKER} -o ex19 ex19.o  ${PETSC_TS_LIB}
	${RM} ex19.o

ex20: ex20.o  chkopts
	-${CLINKER} -o ex20 ex20.o  ${PETSC_TS_LIB}
	${RM} ex20.o

ex20adj: ex20adj.o  chkopts
	-${CLINKER} -o ex20adj ex20adj.o  ${PETSC_TS_LIB}
	${RM} ex20adj.o

ex20fwd: ex20fwd.o  chkopts
	-${CLINKER} -o ex20fwd ex20fwd.o  ${PETSC_TS_LIB}
	${RM} ex20fwd.o

ex20opt_ic: ex20opt_ic.o chkopts
	-${CLINKER} -o ex20opt_ic ex20opt_ic.o ${PETSC_LIB}
	${RM} ex20opt_ic.o

ex20opt_p: ex20opt_p.o chkopts
	-${CLINKER} -o ex20opt_p ex20opt_p.o ${PETSC_LIB}
	${RM} ex20opt_p.o

ex21: ex21.o  chkopts
	-${CLINKER} -o ex21 ex21.o  ${PETSC_TS_LIB}
	${RM} ex21.o

ex22: ex22.o chkopts
	-${CLINKER} -o ex22 ex22.o ${PETSC_TS_LIB}
	${RM} ex22.o

ex22f: ex22f.o  chkopts
	-${FLINKER} -o ex22f ex22f.o ${PETSC_TS_LIB}
	${RM} ex22f.o

ex22f_mf: ex22f_mf.o  chkopts
	-${FLINKER} -o ex22f_mf ex22f_mf.o ${PETSC_TS_LIB}
	${RM} ex22f_mf.o

ex23: ex23.o chkopts
	-${CLINKER} -o ex23 ex23.o ${PETSC_TS_LIB}
	${RM} ex23.o

ex24: ex24.o chkopts
	-${CLINKER} -o ex24 ex24.o ${PETSC_TS_LIB}
	${RM} ex24.o

ex25: ex25.o chkopts
	-${CLINKER} -o ex25 ex25.o ${PETSC_TS_LIB}
	${RM} ex25.o

ex26: ex26.o chkopts
	-${CLINKER} -o ex26 ex26.o ${PETSC_TS_LIB}
	${RM} ex26.o

ex27: ex27.o chkopts
	-${CLINKER} -o ex27 ex27.o ${PETSC_TS_LIB}
	${RM} ex27.o

ex28: ex28.o chkopts
	-${CLINKER} -o ex28 ex28.o ${PETSC_TS_LIB}
	${RM} ex28.o

ex29: ex29.o chkopts
	-${CLINKER} -o ex29 ex29.o ${PETSC_TS_LIB}
	${RM} ex29.o

ex31: ex31.o chkopts
	-${CLINKER} -o ex31 ex31.o ${PETSC_TS_LIB}
	${RM} ex31.o

ex34: ex34.o chkopts
	-${CLINKER} -o ex34 ex34.o ${PETSC_TS_LIB}
	${RM} ex34.o

ex35: ex35.o chkopts
	-${CLINKER} -o ex35 ex35.o ${PETSC_TS_LIB}
	${RM} ex35.o

ex36: ex36.o chkopts
	-${CLINKER} -o ex36 ex36.o ${PETSC_TS_LIB}
	${RM} ex36.o

ex36SE: ex36SE.o chkopts
	-${CLINKER} -o ex36SE ex36SE.o ${PETSC_TS_LIB}
	${RM} ex36SE.o

ex36A: ex36A.o chkopts
	-${CLINKER} -o ex36A ex36A.o ${PETSC_TS_LIB}
	${RM} ex36A.o

extchem: extchem.o chkopts
	-${CLINKER} -o extchem extchem.o ${PETSC_TS_LIB}
	${RM} extchem.o

extchemfield: extchemfield.o chkopts
	-${CLINKER} -o extchemfield extchemfield.o ${PETSC_TS_LIB}
	${RM} extchemfield.o

ex40: ex40.o chkopts
	-${CLINKER} -o ex40 ex40.o ${PETSC_TS_LIB}
	${RM} ex40.o

ex41: ex41.o chkopts
	-${CLINKER} -o ex41 ex41.o ${PETSC_TS_LIB}
	${RM} ex41.o

ex42: ex42.o chkopts
	-${CLINKER} -o ex42 ex42.o ${PETSC_TS_LIB}
	${RM} ex42.o

ex43: ex43.o chkopts
	-${CLINKER} -o ex43 ex43.o ${PETSC_TS_LIB}
	${RM} ex43.o

ex44: ex44.o chkopts
	-${CLINKER} -o ex44 ex44.o ${PETSC_TS_LIB}
	${RM} ex44.o

ex48: ex48.o chkopts
	-${CLINKER} -o ex48 ex48.o ${PETSC_TS_LIB}
	${RM} ex48.o

ex49: ex49.o chkopts
	-${CLINKER} -o ex49 ex49.o ${PETSC_TS_LIB}
	${RM} ex49.o

include ${PETSC_DIR}/lib/petsc/conf/test
