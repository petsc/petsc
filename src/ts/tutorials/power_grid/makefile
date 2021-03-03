-include ../../../../petscdir.mk
CFLAGS          =
FFLAGS          =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/ts/tutorials/power_grid/
EXAMPLESC       = ex1.c ex2.c ex3.c ex3adj.c ex3sa.c ex3opt.c ex3opt_fd.c ex5.c \
                  ex6.c ex7.c ex8.c ex9adj.c ex9.c ex9opt.c
EXAMPLESF       =
EXAMPLESFH      =
MANSEC          = TS
DIRS            = stability_9bus
CLEANFILES      = TS-data-*

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
