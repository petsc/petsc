-include ../../../../petscdir.mk

LOCDIR          = src/ts/tutorials/multirate/
MANSEC          = TS
OBJECTS_FV      = finitevolume1d.o

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex4: ex4.o $(OBJECTS_FV)
ex5: ex5.o $(OBJECTS_FV)
ex6: ex6.o $(OBJECTS_FV)
ex8: ex8.o $(OBJECTS_FV)

include ${PETSC_DIR}/lib/petsc/conf/test
