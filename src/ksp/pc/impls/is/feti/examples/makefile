
# -----------------------------------------------------------------------
# Specify  user-defined compiler flags
# -----------------------------------------------------------------------
CFLAGS    =
# -----------------------------------------------------------------------
# Define application source and examples
# -----------------------------------------------------------------------
#

ALL: main

LOCDIR         = src/ksp/pc/impls/is/feti/examples/
EXAMPLESC      = ex1.c
EXAMPLESO      = ex1.o

ex1: chkopts ${EXAMPLESO}
	-$(RM) ex1
	-$(CLINKER) -o ex1 ${EXAMPLESO} $(PETSC_SLES_LIB)
	-$(RM) ${EXAMPLESO}

run: 
	mpiexec -n 3 ex1 -N 27

# provide input files e.g. fetidp_data.tgz

include $(PETSC_DIR)/conf/base

include $(PETSC_DIR)/conf/test
