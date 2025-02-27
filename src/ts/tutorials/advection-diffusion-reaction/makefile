-include ../../../../petscdir.mk

CLEANFILES      = TS-data-*
OBJECTS_EX5     = reaction_diffusion.o

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex5: $(OBJECTS_EX5)
ex5adj: $(OBJECTS_EX5)
ex5adj_mf: $(OBJECTS_EX5)
ex5opt_ic: $(OBJECTS_EX5)
