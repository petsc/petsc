#
#   This sets PETSC_DIR if it has not been set in the environment; it uses the path of this file, not the path of the makefile that includes this file
#
PETSC_DIR_TMP := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
export PETSC_DIR ?= $(PETSC_DIR_TMP)

