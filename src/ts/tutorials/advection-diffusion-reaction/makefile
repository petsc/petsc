-include ../../../../petscdir.mk

OBJECTS_EX5     = reaction_diffusion.o
CLEANFILES      = TS-data-* ex5adj ex5adj_mf ex5opt_ic ${OBJECTS_EX5}

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex5: $(OBJECTS_EX5)
ex5adj: $(OBJECTS_EX5)
ex5adj_mf: $(OBJECTS_EX5)
ex5opt_ic: $(OBJECTS_EX5)
