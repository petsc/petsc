-include ../petscdir.mk
CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/dm/tutorials/
EXAMPLESC       = ex1.c ex3.c ex4.c ex5.c ex6.c ex7.c ex9.c ex10.c ex12.c ex15.c ex19.c ex20.c ex51.c swarm_ex1.c swarm_ex2.c swarm_ex3.c
EXAMPLESF       = ex11f90.F90 ex13f90aux.F90 ex13f90.F90
MANSEC          = DM
SUBMANSEC       = DMDA
CLEANFILES      = output_file output_file.info swarm_ex[1-3]

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

ex13f90.o: ex13f90aux.o
ex13f90: ex13f90.o ex13f90aux.o
