-include ../../../../petscdir.mk

LOCDIR          = src/ts/tutorials/advection-diffusion-reaction/
EXAMPLESC       = ex1.c ex2.c ex3.c ex4.c ex5.c ex5adj.c ex5opt_ic.c
EXAMPLESF       = shashi.F90
EXAMPLESFH      =
MANSEC          = TS
DIRS            =
CLEANFILES      = TS-data-*
OBJECTS_EX5     = reaction_diffusion.o

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

ex5: ex5.o $(OBJECTS_EX5)
ex5adj: ex5adj.o $(OBJECTS_EX5)
ex5adj_mf: ex5adj_mf.o $(OBJECTS_EX5)
ex5opt_ic: ex5opt_ic.o $(OBJECTS_EX5)
