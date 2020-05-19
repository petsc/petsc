-include ../../../petscdir.mk
CFLAGS          =
FFLAGS          =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/ts/tutorials/
EXAMPLESC       = ex1.c ex2.c ex3.c ex4.c ex5.c ex6.c ex7.c ex8.c \
                  ex9.c ex10.c ex11.c ex12.c ex13.c ex14.c ex15.c ex16.c ex17.c \
                  ex19.c ex20.c ex21.c ex22.c ex24.c ex25.c ex26.c \
                  ex28.c ex31.c ex34.c ex35.cxx extchem.c\
                  ex20adj.c ex20opt_p.c ex20opt_ic.c ex20td \
                  ex40.c ex41.c ex42.c ex48.c ex49.c ex50.c ex52.c \
                  ex16fwd.c
EXAMPLESF       = ex1f.F ex22f.F ex22f_mf.F90
MANSEC          = TS
DIRS            = phasefield advection-diffusion-reaction eimex power_grid network multirate optimal_control
CLEANFILES      = TS-data-*

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
