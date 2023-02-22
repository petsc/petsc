#
#   This file allows the PETSc make commands in makefile to be run if gnumake is the default make without setting the PETSC_DIR.
#   gnumake defaults to using GNUmakefile before makefile

include petscdir.mk

# Target to build (update) the PETSc libraries
all :
	+@$(MAKE) -f makefile --no-print-directory $@

ifeq ($(firstword $(sort 4.1.99 $(MAKE_VERSION))),4.1.99)
include gmakefile
endif

# For any target that doesn't exist in gmakefile, use the legacy makefile (which has the logging features)
% :
	+@$(MAKE) -f makefile --no-print-directory $@
