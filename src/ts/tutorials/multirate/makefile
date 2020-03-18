
CFLAGS          =
FFLAGS          =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/ts/tutorials/multirate/
EXAMPLESC       = ex1.c ex2.c ex3.c ex5.c ex6.c ex7.c
EXAMPLESF       =
EXAMPLESFH      =
MANSEC          = TS
OBJECTS_FV      = finitevolume1d.o

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex5: ex5.o $(OBJECTS_FV)
ex6: ex6.o $(OBJECTS_FV)
ex8: ex8.o $(OBJECTS_FV)

include ${PETSC_DIR}/lib/petsc/conf/test
