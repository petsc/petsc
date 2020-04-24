#
#   This file allows the high level PETSc make commands to be run if gnuumake is the default make without setting the PETSC_DIR.
#   This will prevent errors for the subset of users who run make all without first setting PETSC_DIR which results in wasteful email to the PETSc Team

include petscdir.mk

# Default target
all :
	+@$(MAKE) -f makefile --no-print-directory $@

ifeq ($(firstword $(sort 4.1.99 $(MAKE_VERSION))),4.1.99)
include gmakefile
endif

# For any target that doesn't exist in gmakefile, use the legacy makefile (which has the logging features)
% :
	+@$(MAKE) -f makefile --no-print-directory $@
